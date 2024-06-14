from abc import ABC, abstractmethod

class FindRelevantStrategy(ABC):
    @abstractmethod
    def execute(self, data):
        pass

class FindTopK(FindRelevantStrategy):
    def execute(self, relevance):
        relevance.sort(key = lambda item: item['cos_similarity'], reverse=True)
        top_k = int(input("Please enter the most docs you'd like to match in this query: "))
        top_k_relevant = set()
        for _, entry in enumerate(relevance):
            if len(top_k_relevant) == top_k:
                break
            top_k_relevant.add(entry['doc'])
        return top_k_relevant