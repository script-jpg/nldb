from LLM.LLMInterface import LLMInterface
import openai

class GPT3(LLMInterface):
    def __init__(self):
        self.api_key = None

    def get_response(self, user_message: str, model: str = "gpt-3.5-turbo") -> str:
        if self.api_key is None:
            self.api_key = input("Please enter your OpenAI API key: ")
            openai.api_key = self.api_key

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