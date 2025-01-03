import re
from sentence_transformers import SentenceTransformer, util
from typing import Optional

class ContextSearch:
    def __init__(self, text: Optional[str] = ""):
        self.text = text
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = 0

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
        
        threshold = self.threshold
        result = [sent for sent, score in sentence_scores if score > threshold]
        
        return result


cs = ContextSearch()
cs.threshold = 0.6
with open("entities/base.txt", "r") as file:
    text = file.read()
    cs.text = text

print(cs.advanced_search("Qual meu nome?"))