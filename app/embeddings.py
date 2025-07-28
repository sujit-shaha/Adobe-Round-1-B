from sentence_transformers import SentenceTransformer
import faiss, numpy as np
from typing import List, Dict
import hashlib

class EmbeddingIndex:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dim)
        self.sections: List[Dict] = []
        self._vectors = None
        
        self._cache = {}
        self.cache_hits = 0

    def encode_with_cache(self, texts, batch_size=64, show_progress=False):
        results = []
        texts_to_encode = []
        indices_to_encode = []
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._cache:
                results.append(self._cache[text_hash])
                self.cache_hits += 1
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
            
            for idx, text in enumerate(texts_to_encode):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self._cache[text_hash] = new_embeddings[idx]
        
        final_embeddings = np.zeros((len(texts), self.dim), dtype=np.float32)
        
        result_idx = 0
        for i in range(len(texts)):
            if i in indices_to_encode:
                final_embeddings[i] = new_embeddings[result_idx]
                result_idx += 1
            else:
                text_hash = hashlib.md5(texts[i].encode()).hexdigest()
                final_embeddings[i] = self._cache[text_hash]
                
        return final_embeddings

    def add_sections(self, sections: List[Dict], batch_size: int = 128): 
        texts = [s['content'] for s in sections]
        embeddings = self.encode_with_cache(texts, batch_size, show_progress=False)
        self.index.add(embeddings)
        self.sections.extend(sections)
        self._vectors = embeddings  

    def query(self, query_text: str, top_k: int):
        q_vec = self.model.encode([query_text], convert_to_numpy=True)
        top_k = min(top_k, len(self.sections))
        distances, indices = self.index.search(q_vec, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            sec = self.sections[idx]
            results.append({
                "document": sec["document"],
                "page": sec["page"],
                "section_title": sec["section_title"],
                "content": sec["content"],
                "importance_rank": rank + 1,
                "score": float(distances[0][rank])
            })
        return results
