from abc import ABC, abstractmethod

class LLMInterface(ABC):
    @abstractmethod
    def get_response(self, user_message: str) -> str:
        """
        Abstract method that must be implemented by subclasses.
        :param user_message: The input message from the user.
        :return: The response from the language model.
        """
        pass
