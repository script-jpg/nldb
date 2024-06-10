from LLM.LLMInterface import LLMInterface
import openai

class GPT3(LLMInterface):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def get_response(self, user_message: str, model: str = "gpt-3.5-turbo") -> str:
        chat_completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ]
        )
        return chat_completion.choices[0].message["content"]