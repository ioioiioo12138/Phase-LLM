import os
import httpx
from openai import OpenAI
from retrying import retry
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel


# 这里是你的梯子，如果在国外可以注释掉
os.environ["HTTP_PROXY"] = "127.0.0.1:4780"
os.environ["HTTPS_PROXY"] = "127.0.0.1:4780"


class GPT:
    def __init__(self):
        api_key = ""  # 06
        base_url = ""
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True
            )
        )

    def generate(self, messages):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4,  # 降低温度以提高确定性（范围：0-2，默认1）
            top_k=30,  # 降低top_k以减少随机性（默认50）
            top_p=0.9,
            min_p=0.1
        )
        answer = completion.choices[0].message.content
        if answer == "broken":
            self.generate(messages)
        return answer

    def call(self, content, additional_args=None):
        messages = [{'role': 'user', 'content': content}]
        if additional_args is None:
            additional_args = {}
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=additional_args.get('max_tokens', 8192),
            temperature=additional_args.get('temperature', 0.3),  # 默认0.3以提高确定性
            top_p=additional_args.get('top_p', 0.8)  # 默认0.8以减少随机性
        )
        answer = completion.choices[0].message.content
        if answer == "broken":
            self.call(messages)
        return answer

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    # def retry_call(self, content, additional_args={"max_tokens": 8192}):
    def retry_call(self, content, additional_args=None):
        if additional_args is None:
            additional_args = {"max_completion_tokens": 8192}
        return self.call(content, additional_args)


class DeepSeek:
    def __init__(self, version):
        api_key = "sk-38"
        base_url = "https://api.deepseek.com"

        self.version = version
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        # self.client = OpenAI(
        #     api_key=tencent_key,
        #     base_url=tencent_url,
        # )
    def call(self, content, additional_args=None):
        messages = [{'role': 'user', 'content': content}]
        if additional_args is None:
            additional_args = {}

        completion = self.client.chat.completions.create(
            model=self.version,
            messages=messages,
            max_completion_tokens=8192,
            temperature=additional_args.get('temperature', 0.3),  # 降低温度以提高确定性（默认0.3）
            top_p=additional_args.get('top_p', 0.8)  # 降低top_p以减少随机性（默认0.8）
        )
        answer = completion.choices[0].message.content
        if answer.strip() == "broken":
            self.call(messages)
        return answer

    @retry(wait_fixed=3000, stop_max_attempt_number=3)
    # def retry_call(self, content, additional_args={"max_tokens": 8192}):
    def retry_call(self, content, additional_args=None):
        if additional_args is None:
            additional_args = {"max_completion_tokens": 8190}
        return self.call(content, additional_args)



