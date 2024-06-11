import os
from abc import ABC, abstractmethod


class DataLoader(ABC):

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load_file(self, dir: str, is_selected: callable = None) -> list:
        # Split texts of the file dir into lines, removes lines that is not selected
        pass

    @abstractmethod
    def file_type(self) -> str:
        pass

class MarkdownDataLoader(DataLoader):

    def __init__(self) -> None:
        return None

    def load_file(self, dir: str, is_selected: callable = None) -> list:
        # Split texts of the file dir into lines, removes lines that is not selected

        if is_selected is None:
            is_selected = lambda l: len(l) > 3

        with open(dir, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        lines = [line for line in lines if is_selected(line)]
        return lines
    
    def file_type(self) -> str:
        return 'md'
     
    


if __name__ == "__main__":
    loader = MarkdownDataLoader()
    lines = loader.load_file('./DataFiles/test.md')
    for line in lines:
        print(line)
