from typing import List, Dict, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import os
from pathlib import Path
import re
from dataclasses import dataclass

@dataclass
class SearchResult:
    content: str
    metadata: Dict
    score: float
    
class EnhancedDocManager:
    def __init__(
        self, 
        doc_path: str, 
        api_key: str, 
        cache_dir: str = ".cache",
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize EnhancedDocManager with improved chunking and search capabilities.
        
        Args:
            doc_path: string containing API documentation
            api_key: Google API key for embeddings
            cache_dir: Directory to store cached embeddings and index
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            similarity_threshold: Minimum similarity score for relevant chunks
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        genai.configure(api_key=api_key)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chunks = self._load_and_chunk_docs(doc_path)
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._create_or_load_vector_store()

    def _extract_headers(self, text: str) -> List[Tuple[str, int]]:
        """Extract markdown headers and their positions."""
        headers = []
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
            headers.append((match.group(2).strip(), match.start()))
        return headers

    def _get_section_context(self, position: int, headers: List[Tuple[str, int]]) -> str:
        """Get hierarchical section context for a given position."""
        current_headers = []
        for header, pos in headers:
            if pos > position:
                break
            current_headers.append(header)
        return " > ".join(current_headers[-3:])
    
    def _smart_chunk_text(self, text: str) -> List[Document]:
        """
        Improved chunking that respects markdown structure and maintains context.
        """
        headers = self._extract_headers(text)
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )
        
        chunks = []
        raw_chunks = splitter.create_documents([text])
        
        for i, chunk in enumerate(raw_chunks):
            chunk_start = text.find(chunk.page_content)
            if chunk_start == -1:
                continue
                
            section_context = self._get_section_context(chunk_start, headers)
            
            metadata = {
                "doc_id": i,
                "section_context": section_context,
                "chunk_start": chunk_start,
                "chunk_end": chunk_start + len(chunk.page_content)
            }

            content = chunk.page_content.strip()
            if content:
                chunks.append(Document(page_content=content, metadata=metadata))
        
        return chunks

    def _load_and_chunk_docs(self, doc_path: str) -> List[Document]:
        """Load documents and split into chunks with improved chunking strategy."""
        text = doc_path
        return self._smart_chunk_text(text)

    def _initialize_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Initialize the embeddings model with optimized settings."""
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key,
            task_type="retrieval_document",
            temperature=0.1
        )

    def _create_or_load_vector_store(self) -> faiss.IndexFlatL2:
        """Create or load FAISS index with improved caching."""
        index_path = Path(self.cache_dir) / "faiss_index.bin"
        metadata_path = Path(self.cache_dir) / "chunks_metadata.pkl"
        embeddings_path = Path(self.cache_dir) / "embeddings.npy"
        
        if all(p.exists() for p in [index_path, metadata_path, embeddings_path]):
            index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
            return index
            
        texts = [doc.page_content for doc in self.chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        
        dimension = len(embeddings_list[0])
        index = faiss.IndexFlatL2(dimension)
        embeddings_array = np.array(embeddings_list, dtype=np.float32)
        index.add(embeddings_array)
        
        faiss.write_index(index, str(index_path))
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        np.save(str(embeddings_path), embeddings_array)
        
        return index

    def _calculate_similarity_score(self, distance: float) -> float:
        """Convert FAISS L2 distance to a similarity score."""
        return 1 / (1 + distance)

    def get_relevant_chunks(
        self, 
        query: str, 
        k: int = 5,
        min_score: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Get relevant document chunks with improved retrieval and filtering.
        
        Args:
            query: The search query
            k: Number of chunks to return
            min_score: Minimum similarity score (0-1) to include chunk
            
        Returns:
            List of SearchResult objects containing relevant chunks and metadata
        """
        min_score = min_score or self.similarity_threshold
        
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        k_search = min(k * 2, len(self.chunks))
        distances, indices = self.vector_store.search(query_embedding, k_search)
        
        results = []
        seen_sections = set()  # Track unique sections to avoid redundancy
        
        for distance, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            score = self._calculate_similarity_score(distance)
            
            if score < min_score:
                continue
                
            section = chunk.metadata.get('section_context', '')
            if section in seen_sections:
                continue
                
            seen_sections.add(section)
            
            results.append(SearchResult(
                content=chunk.page_content,
                metadata=chunk.metadata,
                score=score
            ))
            
            if len(results) >= k:
                break
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def format_results(self, results: List[SearchResult], include_scores: bool = False) -> str:
        """Format search results into a readable string."""
        formatted = []
        for result in results:
            content = [f"Section: {result.metadata['section_context']}"]
            if include_scores:
                content.append(f"Relevance Score: {result.score:.2f}")
            content.append("\nContent:")
            content.append(result.content)
            formatted.append("\n".join(content))
        
        return "\n\n---\n\n".join(formatted)