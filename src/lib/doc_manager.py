from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DocChunkManager:
    def __init__(self, api_docs: str, chunk_size: int = 1000, overlap: int = 100):
        self.chunks = self._create_chunks(api_docs, chunk_size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = self.vectorizer.fit_transform([chunk['text'] for chunk in self.chunks])
    
    def _create_chunks(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, str]]:
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            if current_size + len(line) > chunk_size and current_chunk:
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_line': max(0, len(chunks) * (chunk_size - overlap))
                })
                overlap_lines = current_chunk[-3:]
                current_chunk = overlap_lines
                current_size = sum(len(line) for line in overlap_lines)
            
            current_chunk.append(line)
            current_size += len(line)
        
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'start_line': max(0, len(chunks) * (chunk_size - overlap))
            })
        
        return chunks
    
    def get_relevant_chunks(self, query: str, num_chunks: int = 3) -> str:
        query_vector = self.vectorizer.transform([query])
        similarities = np.array(self.chunk_vectors.dot(query_vector.T).toarray()).flatten()
        top_indices = similarities.argsort()[-num_chunks:][::-1]
        relevant_text = "\n\n".join([self.chunks[i]['text'] for i in sorted(top_indices)])
        return relevant_text