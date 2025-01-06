from typing import List, Dict, Optional, Tuple, Set
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import faiss
import numpy as np
import pickle
from pathlib import Path
import re
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    content: str
    metadata: Dict
    score: float

class StructuredDocManager:
    def __init__(
        self, 
        doc_path: str, 
        api_key: str, 
        cache_dir: str = ".cache",
        base_chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.6
    ):
        """
        Initialize StructuredDocManager optimized for API documentation.
        
        Args:
            doc_path: String containing API documentation
            api_key: Google API key for embeddings
            cache_dir: Directory to store cached embeddings and index
            base_chunk_size: Base size for text chunks (will be adjusted based on content)
            chunk_overlap: Overlap between chunks
            similarity_threshold: Minimum similarity score for relevant chunks
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        # genai.configure(api_key=api_key)
        
        self.base_chunk_size = base_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize content stats
        self.content_stats = {
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'header_depth': defaultdict(int)
        }
        
        self.chunks = self._load_and_chunk_docs(doc_path)
        self.embeddings = self._initialize_embeddings()
        self.vector_store = self._create_or_load_vector_store()
        
        logger.info(f"Initialized with {self.content_stats['total_chunks']} chunks")
        logger.info(f"Average chunk size: {self.content_stats['avg_chunk_size']:.2f} characters")
        
    def _extract_code_blocks(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract code blocks and their positions to prevent splitting within them."""
        code_pattern = r'```[\s\S]*?```'
        code_blocks = re.finditer(code_pattern, text)
        placeholders = []
        extracted_blocks = []
        
        for i, match in enumerate(code_blocks):
            placeholder = f"CODE_BLOCK_{i}"
            extracted_blocks.append(match.group(0))
            placeholders.append(placeholder)
            
        processed_text = re.sub(code_pattern, lambda x: placeholders[len(placeholders) - 1], text)
        return processed_text, extracted_blocks

    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restore code blocks to their original positions."""
        restored = text
        for i, block in enumerate(code_blocks):
            restored = restored.replace(f"CODE_BLOCK_{i}", block)
        return restored

    def _get_markdown_headers(self) -> List[Tuple[str, int]]:
        """Define markdown headers for splitting."""
        headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        return headers

    def _clean_chunk_content(self, content: str) -> str:
        """Clean and normalize chunk content."""
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        # Remove trailing/leading whitespace while preserving code blocks
        lines = content.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
            else:
                if in_code_block:
                    cleaned_lines.append(line)  # Preserve code block content exactly
                else:
                    cleaned_lines.append(line.strip())
                    
        return '\n'.join(cleaned_lines).strip()

    def _smart_chunk_text(self, text: str) -> List[Document]:
        """
        Enhanced chunking strategy specifically for API documentation.
        """
        # First extract code blocks to prevent splitting them
        processed_text, code_blocks = self._extract_code_blocks(text)
        
        # Split on headers first
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._get_markdown_headers()
        )
        header_splits = header_splitter.split_text(processed_text)
        
        # Initialize recursive splitter for further splitting of large sections
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.base_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        final_chunks = []
        total_chars = 0
        
        for doc in header_splits:
            # Track header depth statistics
            header_level = len(doc.metadata.get('Header 1', '').split(' > '))
            self.content_stats['header_depth'][header_level] += 1
            
            # Handle large sections
            if len(doc.page_content) > self.base_chunk_size:
                smaller_chunks = recursive_splitter.split_text(doc.page_content)
                for i, chunk in enumerate(smaller_chunks):
                    # Restore code blocks
                    chunk_content = self._restore_code_blocks(chunk, code_blocks)
                    chunk_content = self._clean_chunk_content(chunk_content)
                    
                    metadata = doc.metadata.copy()
                    metadata.update({
                        'chunk_index': i,
                        'is_subsection': True,
                        'original_length': len(doc.page_content)
                    })
                    
                    final_chunks.append(Document(
                        page_content=chunk_content,
                        metadata=metadata
                    ))
                    total_chars += len(chunk_content)
            else:
                # Restore code blocks for smaller sections
                doc_content = self._restore_code_blocks(doc.page_content, code_blocks)
                doc_content = self._clean_chunk_content(doc_content)
                final_chunks.append(Document(
                    page_content=doc_content,
                    metadata=doc.metadata
                ))
                total_chars += len(doc_content)
        
        # Update content statistics
        self.content_stats['total_chunks'] = len(final_chunks)
        self.content_stats['avg_chunk_size'] = total_chars / len(final_chunks) if final_chunks else 0
        
        return final_chunks

    def _load_and_chunk_docs(self, doc_path: str) -> List[Document]:
        """Load documents and split into chunks with improved chunking strategy."""
        return self._smart_chunk_text(doc_path)

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
            logger.info("Loading cached vector store")
            index = faiss.read_index(str(index_path))
            with open(metadata_path, 'rb') as f:
                self.chunks = pickle.load(f)
            return index
            
        logger.info("Creating new vector store")
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
        """Convert FAISS L2 distance to a similarity score with temperature scaling."""
        return 1 / (1 + np.exp(distance))  # Sigmoid scaling for better score distribution

    def _get_context_similarity(self, query_headers: Set[str], chunk_headers: Set[str]) -> float:
        """Calculate similarity between query context and chunk context."""
        if not query_headers or not chunk_headers:
            return 0.0
        intersection = query_headers.intersection(chunk_headers)
        union = query_headers.union(chunk_headers)
        return len(intersection) / len(union)

    def get_relevant_chunks(
        self, 
        query: str, 
        k: int = 5,
        min_score: Optional[float] = None,
        context_boost: float = 0.2  # Weight for context similarity
    ) -> List[SearchResult]:
        """
        Get relevant document chunks with improved retrieval and scoring.
        
        Args:
            query: The search query
            k: Number of chunks to return
            min_score: Minimum similarity score (0-1) to include chunk
            context_boost: Weight for header context similarity in final score
            
        Returns:
            List of SearchResult objects containing relevant chunks and metadata
        """
        min_score = min_score or self.similarity_threshold
        
        # Extract potential header context from query
        query_headers = set(re.findall(r'#\w+|\"([^\"]+)\"', query))
        
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Get more initial results for reranking
        k_search = min(k * 3, len(self.chunks))
        distances, indices = self.vector_store.search(query_embedding, k_search)
        
        results = []
        seen_content = set()
        
        for distance, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            content_hash = hash(chunk.page_content)
            
            if content_hash in seen_content:
                continue
                
            seen_content.add(content_hash)
            
            # Calculate base similarity score
            base_score = self._calculate_similarity_score(distance)
            
            # Calculate context similarity
            chunk_headers = set(
                header for key, value in chunk.metadata.items() 
                if key.startswith('Header') and value
                for header in value.split(' > ')
            )
            context_score = self._get_context_similarity(query_headers, chunk_headers)
            
            # Combine scores
            final_score = base_score * (1 + context_boost * context_score)
            
            if final_score < min_score:
                continue
            
            results.append(SearchResult(
                content=chunk.page_content,
                metadata=chunk.metadata,
                score=final_score
            ))
            
            if len(results) >= k:
                break
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def format_results(self, results: List[SearchResult], include_scores: bool = False) -> str:
        """Format search results into a readable string."""
        formatted = []
        for result in results:
            # Build header context string
            headers = []
            for i in range(1, 5):  # Support up to 4 levels of headers
                header_key = f'Header {i}'
                if header_key in result.metadata:
                    headers.append(result.metadata[header_key])
            
            content = [f"Section: {' > '.join(filter(None, headers))}"]
            
            if include_scores:
                content.append(f"Relevance Score: {result.score:.3f}")
            
            content.append("\nContent:")
            content.append(result.content)
            formatted.append("\n".join(content))
        
        return "\n\n---\n\n".join(formatted)

    def get_statistics(self) -> Dict:
        """Get statistics about the document chunks and headers."""
        return {
            'total_chunks': self.content_stats['total_chunks'],
            'average_chunk_size': self.content_stats['avg_chunk_size'],
            'header_depth_distribution': dict(self.content_stats['header_depth'])
        }