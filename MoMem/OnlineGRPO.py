from contextlib import nullcontext
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


DEFAULT_PROMPT_FIELDS = (
    "messages",
    "prompt",
    "question",
    "problem",
)
DEFAULT_ANSWER_FIELDS = (
    "gold_answer",
    "answer",
    "solution",
    "final_answer",
)


def round_up_to_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def build_generation_kwargs(presence_penalty):
    generation_kwargs = {}
    if presence_penalty != 0.0:
        generation_kwargs["presence_penalty"] = presence_penalty
    return generation_kwargs or None


class OnlineGRPO:
    def __init__(
        self,
        base_model,
        lora_path,
        reward_func=None,
        reward_name=None,
        system_prompt="",
        training_samples=None,
        prompt_fields=None,
        answer_fields=None,
        extra_training_fields=None,
        require_gold_answer=False,
        learning_rate=1e-5,
        num_generations=16,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_completion_length=4096,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=None,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        vllm_length_buffer=256,
        vllm_max_model_length=None,
        vllm_tensor_parallel_size=1,
        vllm_enable_sleep_mode=False,
        enable_thinking=False,
        vllm_gpu_memory_utilization=0.6,
        reward_mode=None,
        resume_lora_path=None,
    ):
        if isinstance(reward_func, str) and reward_mode is None:
            reward_mode = reward_func
            reward_func = None
        if reward_func is None:
            raise ValueError(
                "OnlineGRPO does not provide built-in reward modes. "
                "Pass a dataset-specific reward_func to OnlineGRPO(reward_func=...)."
            )
        if not callable(reward_func):
            raise TypeError(f"reward_func must be callable; got {type(reward_func).__name__}.")

        self.base_model = base_model
        self.lora_path = lora_path
        self.reward_func = reward_func
        self.reward_name = reward_name or reward_mode or getattr(reward_func, "__name__", "reward_func")
        self.reward_mode = self.reward_name
        self.system_prompt = system_prompt
        self.prompt_fields = tuple(prompt_fields or DEFAULT_PROMPT_FIELDS)
        self.answer_fields = tuple(answer_fields or DEFAULT_ANSWER_FIELDS)
        self.extra_training_fields = tuple(extra_training_fields or ())
        self.require_gold_answer = require_gold_answer
        self.enable_thinking = enable_thinking
        self.max_completion_length = max_completion_length
        self.learning_rate = learning_rate
        self.num_generations = num_generations
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.vllm_length_buffer = vllm_length_buffer
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_enable_sleep_mode = vllm_enable_sleep_mode
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.resume_lora_path = resume_lora_path

        self.lora_config = self._build_lora_config()
        self.tokenizer = self._build_tokenizer()
        self.vllm_max_model_length = self.resolve_vllm_max_model_length(
            training_samples=training_samples,
            vllm_max_model_length=vllm_max_model_length,
        )
        print(self.vllm_max_model_length)

        self.train_config = self._build_train_config()
        self.trainer = self._build_trainer()
        self.model = self.trainer.model
        self._last_synced_vllm_step = None
        self.peft_config = None

    def _build_lora_config(self):
        return LoraConfig(
            r=8,
            lora_alpha=64,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )

    def _build_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _build_train_config(self):
        return GRPOConfig(
            output_dir=self.lora_path,
            learning_rate=self.learning_rate,
            lr_scheduler_type="cosine",
            num_generations=self.num_generations,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_completion_length=self.max_completion_length,
            tf32=True,
            bf16=True,
            beta=0.0,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            num_train_epochs=1,
            repetition_penalty=self.repetition_penalty,
            generation_kwargs=build_generation_kwargs(self.presence_penalty),
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            vllm_max_model_length=self.vllm_max_model_length,
            vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
            vllm_enable_sleep_mode=self.vllm_enable_sleep_mode,
            vllm_importance_sampling_correction=True,
            vllm_importance_sampling_cap=2.0,
            gradient_checkpointing=True,
            use_liger_kernel=True,
            report_to="none",
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            model_init_kwargs=self._build_model_init_kwargs(),
            logging_steps=1,
            log_completions=True,
            num_completions_to_print=0,
            log_unique_prompts=False,
        )

    def _build_model_init_kwargs(self):
        return {
            "device_map": None,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        }

    def _build_model(self):
        if self.resume_lora_path is None:
            return self.base_model

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **self._build_model_init_kwargs(),
        )
        return PeftModel.from_pretrained(
            base_model,
            self.resume_lora_path,
            is_trainable=True,
        )

    def _build_trainer(self):
        model = self._build_model()
        return GRPOTrainer(
            model=model,
            reward_funcs=[self.reward_func],
            args=self.train_config,
            train_dataset=self._build_bootstrap_dataset(),
            processing_class=self.tokenizer,
            peft_config=None if self.resume_lora_path is not None else self.lora_config,
        )

    def _build_bootstrap_dataset(self):
        sample = self._build_training_sample({"prompt": "boost prompt", "gold_answer": ""})
        return Dataset.from_list([sample])

    def _set_trainer_output_dir(self, output_dir):
        output_dir = str(output_dir)
        self.train_config.output_dir = output_dir
        self.trainer.args.output_dir = output_dir

    def resolve_vllm_max_model_length(
        self,
        training_samples=None,
        vllm_max_model_length=None,
    ):
        if vllm_max_model_length is not None:
            return vllm_max_model_length

        samples = training_samples or [{"prompt": "boost prompt", "gold_answer": ""}]
        max_prompt_length = 0
        for sample in samples:
            training_sample = self._build_training_sample(sample)
            prompt_ids = self.tokenizer.apply_chat_template(
                training_sample["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            max_prompt_length = max(max_prompt_length, len(prompt_ids))

        estimated_length = (
            max_prompt_length + self.max_completion_length + self.vllm_length_buffer
        )
        return round_up_to_multiple(estimated_length, 256)

    def _build_training_prompt(self, messages):
        if isinstance(messages, str):
            user_content = messages.strip()
            if not user_content:
                raise ValueError("Cannot build a prompt from an empty implicit-memory sample.")

            prompt = []
            if self.system_prompt:
                prompt.append({"role": "system", "content": self.system_prompt})
            prompt.append({"role": "user", "content": user_content})
            return prompt

        if not isinstance(messages, list):
            raise TypeError(
                "OnlineGRPO.update expects a string, message list, or training-sample dict; "
                f"got {type(messages).__name__}."
            )

        if not messages:
            raise ValueError("Cannot build a prompt from an empty implicit-memory sample.")

        return messages

    def _get_first_present_field(self, sample, field_names):
        for field_name in field_names:
            value = sample.get(field_name)
            if value is not None:
                return value
        return None

    def _build_training_sample(self, sample):
        gold_answer = None

        if isinstance(sample, dict):
            prompt_source = self._get_first_present_field(sample, self.prompt_fields)
            if prompt_source is None:
                raise ValueError(
                    "Training sample is missing a prompt field; configured prompt_fields="
                    f"{self.prompt_fields}."
                )
            gold_answer = self._get_first_present_field(sample, self.answer_fields)
        else:
            prompt_source = sample

        training_sample = {"prompt": self._build_training_prompt(prompt_source)}

        if gold_answer is not None:
            training_sample["gold_answer"] = str(gold_answer).strip()
        elif self.require_gold_answer:
            raise ValueError("The current reward_func requires gold_answer.")

        if isinstance(sample, dict):
            for field_name in self.extra_training_fields:
                value = sample.get(field_name)
                if value is not None:
                    training_sample[field_name] = str(value)

        return training_sample

    def _is_message_list(self, value):
        return (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) and "role" in item for item in value)
        )

    def _build_training_dataset(self, samples):
        if isinstance(samples, (list, tuple)) and not self._is_message_list(samples):
            if not samples:
                raise ValueError("Cannot update OnlineGRPO with an empty training-sample list.")
            training_samples = [self._build_training_sample(sample) for sample in samples]
        else:
            training_samples = [self._build_training_sample(samples)]
        return Dataset.from_list(training_samples)

    def update_and_save(self, messages, lora_path=None):
        target_lora_path = Path(lora_path or self.lora_path)
        target_lora_path.mkdir(parents=True, exist_ok=True)
        self._set_trainer_output_dir(target_lora_path)
        self.trainer.train_dataset = self._build_training_dataset(messages)
        self.trainer.train()
        self.trainer.save_model(str(target_lora_path))
        
    def update(self, messages, lora_path=None):
        target_lora_path = Path(lora_path or self.lora_path)
        target_lora_path.mkdir(parents=True, exist_ok=True)
        self._set_trainer_output_dir(target_lora_path)
        self.trainer.train_dataset = self._build_training_dataset(messages)
        self.trainer.train()
        self._sync_vllm_weights()
        
    def update_many(self, samples, lora_path=None):
        self.update(samples, lora_path=lora_path)

    def _sync_vllm_weights(self):
        backend = getattr(self.trainer, "vllm_generation", None)
        if backend is None:
            return
        global_step = getattr(self.trainer.state, "global_step", None)
        if self._last_synced_vllm_step == global_step:
            return
        backend.sync_weights()
        self._last_synced_vllm_step = global_step

    def _normalize_generation_prompts(self, prompts):
        if isinstance(prompts, (str, dict)) or self._is_message_list(prompts):
            return [prompts]
        if not isinstance(prompts, (list, tuple)):
            raise TypeError(f"prompts must be a string, message list, or list; got {type(prompts).__name__}.")
        return prompts

    def _build_generation_prompt_ids(self, prompts):
        prompt_ids = []
        for prompt in prompts:
            if isinstance(prompt, dict):
                prompt = self._build_training_sample(prompt)["prompt"]
            else:
                prompt = self._build_training_prompt(prompt)
            prompt_ids.append(
                self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )
        return prompt_ids

    def generate(
        self,
        prompts,
        batch_size=1,
        max_new_tokens=None,
        temperature=0.7,
        topk=20,
        top=0.8,
        min_p=0.0,
    ):
        prompts = self._normalize_generation_prompts(prompts)

        if not prompts:
            return []

        backend = getattr(self.trainer, "vllm_generation", None)
        if backend is None:
            raise RuntimeError("The current trainer does not have a vLLM generation backend.")
        self._sync_vllm_weights()

        max_new_tokens = max_new_tokens or self.max_completion_length
        batch_size = max(1, int(batch_size))
        generated_texts = []

        original_temperature = backend.temperature
        original_top_p = backend.top_p
        original_top_k = backend.top_k
        original_min_p = getattr(backend, "min_p", None)
        original_max_completion_length = backend.max_completion_length

        backend.temperature = temperature
        backend.top_p = top
        backend.top_k = topk
        if original_min_p is not None:
            backend.min_p = min_p
        backend.max_completion_length = max_new_tokens

        try:
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                prompt_ids = self._build_generation_prompt_ids(batch_prompts)

                _, completion_ids, _, _ = backend.generate(
                    prompts=prompt_ids,
                    images=None,
                    num_generations=1,
                )
                generated_texts.extend(
                    self.tokenizer.decode(ids, skip_special_tokens=True).strip()
                    for ids in completion_ids
                )
        finally:
            backend.temperature = original_temperature
            backend.top_p = original_top_p
            backend.top_k = original_top_k
            if original_min_p is not None:
                backend.min_p = original_min_p
            backend.max_completion_length = original_max_completion_length

        return generated_texts

    def generate_base(
        self,
        prompts,
        batch_size=1,
        max_new_tokens=None,
        temperature=0.7,
        topk=20,
        top=0.8,
        min_p=0.0,
    ):
        prompts = self._normalize_generation_prompts(prompts)

        if not prompts:
            return []

        model = self.trainer.model
        max_new_tokens = max_new_tokens or self.max_completion_length
        batch_size = max(1, int(batch_size))
        generated_texts = []

        was_training = model.training
        model.eval()

        adapter_context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        try:
            with adapter_context, torch.no_grad():
                for start in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[start : start + batch_size]
                    prompt_ids = self._build_generation_prompt_ids(batch_prompts)
                    inputs = self.tokenizer.pad(
                        [{"input_ids": ids} for ids in prompt_ids],
                        padding=True,
                        return_tensors="pt",
                    )
                    device = next(model.parameters()).device
                    inputs = {key: value.to(device) for key, value in inputs.items()}

                    generation_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id,
                    }
                    if temperature is not None and temperature > 0:
                        generation_kwargs.update(
                            {
                                "do_sample": True,
                                "temperature": temperature,
                                "top_k": topk,
                                "top_p": top,
                            }
                        )
                        if min_p is not None and min_p > 0:
                            generation_kwargs["min_p"] = min_p
                    else:
                        generation_kwargs["do_sample"] = False

                    output_ids = model.generate(**inputs, **generation_kwargs)
                    prompt_length = inputs["input_ids"].shape[1]
                    generated_texts.extend(
                        self.tokenizer.decode(ids[prompt_length:], skip_special_tokens=True).strip()
                        for ids in output_ids
                    )
        finally:
            if was_training:
                model.train()

        return generated_texts
