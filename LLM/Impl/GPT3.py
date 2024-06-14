from LLM.LLMInterface import LLMInterface
from openai import OpenAI

class GPT3(LLMInterface):
    def __init__(self, openapi_key = None):
        if openapi_key is None:
            openapi_key = input("Please enter your OpenAI API key: ")
            self.client = OpenAI(api_key=openapi_key)

    def get_response(self, user_message: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            model="gpt-3.5-turbo",
        )
        content = chat_completion.choices[0].message.content
        return content