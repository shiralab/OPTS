import os
import time

import ollama
import openai
from vllm import LLM, SamplingParams


def setup_llm(type="ollama", name="phi3", seed=None):
    type = type.lower()
    if type == "vllm":
        return VLLM(model_name=name)

    elif type == "ollama":
        return Ollama(model_name=name)

    elif type == "openai":
        return Openai(model_name=name, seed=seed)


class Model:
    def __init__(self, model_name):
        self.model_name: str = model_name

    def query(self, requests):
        pass


class Ollama(Model):
    def __init__(self, model_name):
        print(f"ollama-{model_name}")
        super().__init__(model_name)

    def query(self, requests, sys_request="", **param):
        if isinstance(requests, str):
            requests = [requests]

        outputs = []
        for request in requests:
            if sys_request == "":
                model_request = [{"role": "user", "content": request}]
            else:
                model_request = [
                    {"role": "system", "content": sys_request},
                    {"role": "user", "content": request},
                ]

            output = ollama.chat(
                model=self.model_name, messages=model_request, options=param
            )
            outputs.append(output["message"]["content"].strip())

        return outputs


class Openai(Model):
    def __init__(self, model_name="gpt-4o-mini", seed=None):
        super().__init__(model_name)
        self.seed = seed
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def query(self, requests, sys_request="", **param):
        if isinstance(requests, str):
            requests = [requests]

        outputs = []
        for request in requests:
            backoff_second = 1
            retried = 0
            while True:
                try:
                    if sys_request == "":
                        model_request = {
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": request}],
                            **param,
                        }
                    else:
                        print("system request true")
                        model_request = {
                            "model": self.model_name,
                            "messages": [
                                {"role": "system", "content": sys_request},
                                {"role": "user", "content": request},
                            ],
                            **param,
                        }
                    if self.seed is not None:
                        output = openai.chat.completions.create(
                            seed=self.seed, **model_request
                        )
                    else:
                        output = openai.chat.completions.create(**model_request)

                    outputs.append(output.choices[0].message.content)
                    break
                except Exception as e:
                    error = str(e)
                    print("retring...", error)
                    backoff_second *= 1.2
                    retried = retried + 1
                    if retried > 100:
                        raise
                    else:
                        time.sleep(backoff_second)

        return outputs


class VLLM(Model):
    def __init__(self, model_name):
        super().__init__(model_name)

        self.model = LLM(
            model=model_name,
            dtype="auto",
            gpu_memory_utilization=0.8,
            enable_prefix_caching=True,
        )

        self.tokenizer = self.model.get_tokenizer()

    def query(self, requests, sys_request="", **param):
        if isinstance(requests, str):
            requests = [requests]

        chat_requests = []
        for request in requests:
            if sys_request == "":
                chat_request = [{"role": "user", "content": request}]
                chat_request = self.tokenizer.apply_chat_template(
                    chat_request, tokenize=False, add_generation_prompt=True
                )
                chat_requests.append(chat_request)
            else:
                chat_request = [
                    {"role": "system", "content": sys_request},
                    {"role": "user", "content": request},
                ]
                chat_request = self.tokenizer.apply_chat_template(
                    chat_request, tokenize=False, add_generation_prompt=True
                )
                chat_requests.append(chat_request)

        responses = self.model.generate(
            chat_requests, SamplingParams(**param), use_tqdm=False
        )

        responses = [response.outputs[0].text.strip() for response in responses]

        # gc.collect()
        # torch.cuda.empty_cache()

        return responses
