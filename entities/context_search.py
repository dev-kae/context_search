import re
from sentence_transformers import SentenceTransformer, util

class ContextSearch:
    def __init__(self, text):
        self.text = text
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    @property
    def tokens(self):
        return re.findall(r"\w+|[.,!?;]", self.text)

    @property
    def sentences(self):
        return re.split(r"(?<=[.!?])\s+", self.text)

    def search(self, query):
        return [
            sent
            for sent in self.sentences
            if re.search(rf"\b{query}\b", sent, re.IGNORECASE)
        ]

    def advanced_search(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        sentence_embeddings = self.model.encode(self.sentences, convert_to_tensor=True)
        
        similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]
        
        sentence_scores = list(zip(self.sentences, similarities))
        sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        
        threshold = 0.3
        result = [sent for sent, score in sentence_scores if score > threshold]
        
        return result
