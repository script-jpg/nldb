from transformers import BertTokenizer, BertModel
import torch
from abc import ABC, abstractmethod
from .constants import CONTENT_LINE_EMBEDDING_IDX
from .find_relevant_strategy import FindTopK


class Transformer(ABC):

    @abstractmethod
    def __init__() -> None:
        return None
    
    @abstractmethod
    def name(self) -> str:
        # Return the model name
        pass

    @abstractmethod
    def load_model(self, model_type: str) -> None:
        # Load pre-trained model
        pass


    @abstractmethod
    def encoded(self, msg: str|list) -> list:
        # Encode msg with the model
        pass
    
    @abstractmethod
    def cos_similarity(self, msg1, msg2):
        pass

    @abstractmethod
    def find_relevant(self, query: str, utils: dict):
        pass

class BertTransformer:

    def __init__(self, model_type = 'bert-base-uncased') -> None:
        self.model_type = model_type
        self.load_model(model_type)
        self.cos = torch.nn.CosineSimilarity(0)
        self.strategy = FindTopK()

    def name(self) -> str:
        # Return the model name
        return self.model_type.replace('-', '_')

    def load_model(self, model_type = 'bert-base-uncased') -> None:
        # Load pre-trained model (weights) and pre-trained model tokenizer (vocabulary)
        self.model = BertModel.from_pretrained(model_type)
        self.tokenizer = BertTokenizer.from_pretrained(model_type)

    def encoded(self, msg):
        if isinstance(msg, str):
            # Handle a single string
            inputs = self.tokenizer(msg, return_tensors='pt')
            # Forward pass, get hidden states
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the embeddings of the last hidden state
            last_hidden_states = outputs.last_hidden_state

            # Typically, we use the embedding of the [CLS] token
            sentence_embedding = last_hidden_states[:, 0, :].squeeze()

            return sentence_embedding
        elif isinstance(msg, list):
            # Handle a list of strings
            inputs = self.tokenizer(msg, return_tensors='pt', padding=True, truncation=True)
            # Forward pass, get hidden states
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the embeddings of the last hidden state
            last_hidden_states = outputs.last_hidden_state

            # Typically, we use the embedding of the [CLS] token
            sentence_embeddings = last_hidden_states[:, 0, :]

            return sentence_embeddings

    def cos_similarity(self, msg1, msg2):
        return self.cos(self.encoded(msg1), self.encoded(msg2))
    
    def find_relevant(self, query: str, utils: dict):
        query_vector = self.encoded(query)

        relevance = []
        for doc, util in utils.items():
            for line_data in util['content']:
                line_embedding = line_data[CONTENT_LINE_EMBEDDING_IDX]
                relevance.append({
                    'doc': doc,
                    'cos_similarity': self.cos(query_vector, torch.Tensor(line_embedding)).item()
                })

        
        return self.strategy.execute(relevance)



if __name__ == "__main__":
    similarity_model = BertTransformer()
    sentence1 = "How is AI relevant to healthcare?"
    sentence2 = "Artificial Intelligence (AI) is revolutionizing the healthcare industry by enhancing diagnostic accuracy, personalizing patient care, and optimizing operational efficiency. AI-powered tools such as machine learning algorithms and neural networks can analyze vast amounts of medical data to identify patterns and predict outcomes with remarkable precision. For instance, AI-driven diagnostic systems can detect early signs of diseases like cancer from medical imaging, often surpassing human accuracy. Moreover, personalized treatment plans developed through AI analysis of patient data ensure more effective and tailored healthcare solutions. This transformation is not only improving patient outcomes but also reducing the burden on healthcare professionals, allowing them to focus more on patient interaction and care. As AI continues to evolve, its integration into healthcare promises a future of more efficient, accurate, and patient-centric medical services."
    sentence3 = "The resurgence of electric vehicles (EVs) marks a significant shift in the automotive industry towards sustainable transportation. With advancements in battery technology, improved charging infrastructure, and increased environmental awareness, EVs are becoming more accessible and practical for everyday use. Companies like Tesla, Nissan, and Chevrolet are leading the charge, offering a range of models that cater to various market segments. Government incentives and regulations aimed at reducing carbon emissions are also accelerating the adoption of EVs. This shift is not only reducing our dependence on fossil fuels but also paving the way for cleaner urban environments and a reduction in global greenhouse gas emissions. As the technology continues to develop, with innovations such as solid-state batteries and autonomous driving, the future of transportation is set to be increasingly electric, promising a more sustainable and efficient way to travel."

    sentences = [sentence1, sentence2, sentence3]
    for sentence in sentences:
        print(similarity_model.encoded(sentence).tolist())
    
