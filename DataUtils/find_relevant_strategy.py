from abc import ABC, abstractmethod

class FindRelevantStrategy(ABC):
    @abstractmethod
    def execute(self, data):
        pass

class FindTopK(FindRelevantStrategy):
    def execute(self, relevance):
        relevance.sort(key = lambda item: item['cos_similarity'], reverse=True)
        top_k = int(input("Please enter the most docs you'd like to match in this query: "))
        relevance = relevance[:top_k]
        return set([item['doc'] for item in relevance])