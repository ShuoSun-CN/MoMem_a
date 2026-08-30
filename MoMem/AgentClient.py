import os
import time

import requests
from openai import OpenAI
from transformers import AutoTokenizer


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = "gpt-5.4"

LOCAL_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8090")
LOCAL_MODEL = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


class AgentClient:
    def __init__(
        self,
        backend="remote",
        api_key=None,
        base_url=None,
        model_name=None,
        request_timeout=300,
        max_retries=3,
    ):
        if backend in {"local", "vllm", "local_vllm"}:
            self.backend = "local_vllm"
        elif backend in {"remote", "openai", "remote_openai"}:
            self.backend = "openai"
        else:
            raise ValueError(
                "Unsupported backend. Use 'remote'/'openai' for OpenAI or "
                "'local'/'vllm' for local vLLM."
            )

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.trust_env = False
        self.loaded_lora_name = None

        if self.backend == "local_vllm":
            self.api_key = None
            self.base_url = (base_url or LOCAL_BASE_URL).rstrip("/").removesuffix("/v1")
            self.model_name = model_name or LOCAL_MODEL
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self.client = None
            return

        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI backend requires OPENAI_API_KEY or api_key.")

        self.base_url = (base_url or OPENAI_BASE_URL).rstrip("/")
        self.model_name = OPENAI_MODEL
        self.tokenizer = None
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_models(self):
        try:
            if self.backend == "local_vllm":
                print(f"Model name: {self.model_name}")
                return [self.model_name]

            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            for model_name in model_names:
                print(f"Model name: {model_name}")
            return model_names
        except Exception as e:
            print(f"Failed to fetch models: {e}")
            return []

    @staticmethod
    def _extract_responses_text(response):
        if hasattr(response, "output_text") and response.output_text:
            return str(response.output_text).strip()

        parts = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text = getattr(content, "text", None)
                    if text:
                        parts.append(str(text))
        return "".join(parts).strip()

    def load_lora(
        self,
        lora_path,
        lora_name="implicit_memory_lora_adapter",
        load_inplace=True,
        max_retries=5,
        retry_delay=1.0,
    ):
        if self.backend != "local_vllm":
            raise ValueError("Only the local_vllm backend supports dynamic LoRA loading.")
        if not lora_path:
            raise ValueError("load_lora requires lora_path.")

        for attempt in range(1, max_retries + 1):
            load_resp = self.session.post(
                f"{self.base_url}/v1/load_lora_adapter",
                json={
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                    "load_inplace": load_inplace,
                },
                timeout=self.request_timeout,
            )
            if load_resp.ok:
                self.loaded_lora_name = lora_name
                return lora_name

            if load_resp.status_code != 404 or attempt == max_retries:
                load_resp.raise_for_status()

            time.sleep(retry_delay)

    def unload_lora(self, lora_name=None, ignore_missing=False):
        if self.backend != "local_vllm":
            raise ValueError("Only the local_vllm backend supports dynamic LoRA unloading.")

        target_lora_name = lora_name or self.loaded_lora_name
        if not target_lora_name:
            raise ValueError("unload_lora requires lora_name or a previously loaded LoRA.")

        unload_resp = self.session.post(
            f"{self.base_url}/v1/unload_lora_adapter",
            json={"lora_name": target_lora_name},
            timeout=self.request_timeout,
        )
        if ignore_missing and unload_resp.status_code == 404:
            if self.loaded_lora_name == target_lora_name:
                self.loaded_lora_name = None
            return False

        unload_resp.raise_for_status()
        if self.loaded_lora_name == target_lora_name:
            self.loaded_lora_name = None
        return True

    def chat_completion(
        self,
        prompt,
        system_prompt="",
        print_response=False,
        use_lora=False,
        lora_path=None,
        n=1,
        max_tokens=None,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
    ):
        for attempt in range(1, self.max_retries + 1):
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                if self.backend == "local_vllm":
                    contents = self._complete_local_vllm(
                        messages=messages,
                        use_lora=use_lora,
                        lora_path=lora_path,
                        n=n,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        presence_penalty=presence_penalty,
                    )
                else:
                    contents = self._complete_openai(
                        messages=messages,
                        n=n,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                content = contents[0] if n == 1 else contents
                if print_response:
                    print("AI response:", content)
                return content
            except Exception as e:
                if attempt < self.max_retries:
                    print(f"Chat failed; retry {attempt}: {e}")
                    time.sleep(min(2 ** (attempt - 1), 4))
                else:
                    print(f"Chat failed: {e}")

        return None

    def _complete_local_vllm(
        self,
        messages,
        use_lora=False,
        lora_path=None,
        n=1,
        max_tokens=None,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
    ):
        request_model = self.model_name
        if use_lora:
            if lora_path:
                request_model = self.load_lora(lora_path=lora_path)
            elif self.loaded_lora_name:
                request_model = self.loaded_lora_name
            else:
                raise ValueError("use_lora=True requires lora_path or a previously loaded LoRA.")

        request_payload = {
            "model": request_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "n": n,
        }
        if min_p is not None:
            request_payload["min_p"] = min_p
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json=request_payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return [
            (choice["message"].get("content") or "").strip()
            for choice in response.json()["choices"]
        ]

    def _complete_openai(
        self,
        messages,
        n=1,
        max_tokens=None,
        temperature=0.0,
    ):
        if n != 1:
            raise ValueError("OpenAI Responses API path supports n=1 only.")

        kwargs = {
            "model": self.model_name,
            "input": messages,
            "temperature": temperature,
            "timeout": self.request_timeout,
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        response = self.client.responses.create(**kwargs)
        return [self._extract_responses_text(response)]

    def complete(
        self,
        prompt,
        system_prompt="",
        print_response=False,
        use_lora=False,
        lora_path=None,
        n=1,
        max_tokens=None,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
    ):
        return self.chat_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            print_response=print_response,
            use_lora=use_lora,
            lora_path=lora_path,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
        )


if __name__ == "__main__":
    client = AgentClient()
    client.get_models()

    while True:
        user_input = input("Enter a prompt (type 'exit' to quit): ").strip()
        if user_input == "exit":
            break
        client.chat_completion(
            user_input,
            system_prompt="You are a helpful AI assistant.",
            print_response=True,
        )
