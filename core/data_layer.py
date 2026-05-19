"""
EVIRAG Data Layer
Handles PDF parsing, chunking, embedding, indexing, and figure extraction
"""

import contextlib
import os
import json
import pickle
import hashlib
import sys
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

from config import (
    CORPUS_DIR, VECTOR_STORE_DIR, CACHE_DIR, FIGURES_DIR,
    EMBEDDINGS_DIR, EMBEDDING_CONFIG, VECTOR_STORE_CONFIG, PDF_CONFIG
)
from claim_graph import ClaimGraphStore


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _can_render_live_progress() -> bool:
    """Only emit progress bars when attached to a real terminal."""
    stream = getattr(sys, "stderr", None)
    try:
        return bool(stream) and stream.isatty()
    except Exception:
        return False


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    section: Optional[str] = None
    page_num: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Figure:
    """Represents an extracted figure from a PDF"""
    figure_id: str
    doc_id: str
    page_num: int
    caption: str
    image_path: Path
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    referenced_in_text: List[str] = None
    
    def __post_init__(self):
        if self.referenced_in_text is None:
            self.referenced_in_text = []


class PDFProcessor:
    """Process PDFs: extract text, figures, metadata"""
    
    def __init__(self):
        self.section_headers = [h.lower() for h in PDF_CONFIG["section_headers"]]
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Basic metadata
            meta = {
                "filename": pdf_path.name,
                "title": metadata.get("title", pdf_path.stem),
                "author": metadata.get("author", "Unknown"),
                "year": self._extract_year(metadata),
                "pages": len(doc),
                "path": str(pdf_path)
            }
            
            doc.close()
            return meta
        except Exception as e:
            print(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                "filename": pdf_path.name,
                "title": pdf_path.stem,
                "author": "Unknown",
                "year": None,
                "pages": 0,
                "path": str(pdf_path)
            }
    
    def _extract_year(self, metadata: Dict) -> Optional[int]:
        """Try to extract publication year from metadata"""
        # Simple heuristic - look for 4-digit year in creation/mod date
        for key in ["creationDate", "modDate"]:
            if key in metadata:
                date_str = metadata[key]
                # Try to find YYYY pattern
                import re
                match = re.search(r'(19|20)\d{2}', str(date_str))
                if match:
                    return int(match.group())
        return None
    
    def extract_text_with_sections(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract text with section awareness"""
        try:
            doc = fitz.open(pdf_path)
            sections = []
            current_section = "unknown"
            current_text = []
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                lines = text.split('\n')
                
                for line in lines:
                    line_lower = line.lower().strip()
                    
                    # Check if this is a section header
                    is_header = False
                    for header in self.section_headers:
                        if line_lower.startswith(header) and len(line_lower) < 50:
                            # Save previous section
                            if current_text:
                                sections.append({
                                    "section": current_section,
                                    "text": "\n".join(current_text),
                                    "page_num": page_num
                                })
                                current_text = []
                            
                            current_section = header
                            is_header = True
                            break
                    
                    if not is_header and line.strip():
                        current_text.append(line.strip())
            
            # Add final section
            if current_text:
                sections.append({
                    "section": current_section,
                    "text": "\n".join(current_text),
                    "page_num": page_num
                })
            
            doc.close()
            return sections
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return []
    
    def extract_figures(self, pdf_path: Path, doc_id: str) -> List[Figure]:
        """Extract figures and captions from PDF"""
        if not PDF_CONFIG["extract_figures"]:
            return []
        
        try:
            doc = fitz.open(pdf_path)
            figures = []
            
            for page_num, page in enumerate(doc):
                # Get images
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Extract image
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save image
                    figure_id = f"{doc_id}_p{page_num}_f{img_index}"
                    image_path = FIGURES_DIR / f"{figure_id}.png"
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Try to find caption (heuristic: text near image)
                    caption = self._extract_caption(page, img_index)
                    
                    # Get bounding box (approximate)
                    bbox = (0, 0, 1, 1)  # Placeholder
                    
                    figures.append(Figure(
                        figure_id=figure_id,
                        doc_id=doc_id,
                        page_num=page_num,
                        caption=caption,
                        image_path=image_path,
                        bbox=bbox
                    ))
                    
                    if len(figures) >= PDF_CONFIG["max_figures_per_paper"]:
                        break
                
                if len(figures) >= PDF_CONFIG["max_figures_per_paper"]:
                    break
            
            doc.close()
            return figures
            
        except Exception as e:
            print(f"Error extracting figures from {pdf_path}: {e}")
            return []
    
    def _extract_caption(self, page, img_index: int) -> str:
        """Heuristic caption extraction"""
        # Simplified: just look for "Figure X" patterns
        text = page.get_text()
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if f"figure {img_index}" in line.lower() or f"fig. {img_index}" in line.lower():
                # Return this line and next few
                caption_lines = lines[i:min(i+3, len(lines))]
                return " ".join(caption_lines)
        
        return f"Figure {img_index}"


class TextChunker:
    """Chunk text into semantically meaningful units"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or EMBEDDING_CONFIG["chunk_size"]
        self.overlap = overlap or EMBEDDING_CONFIG["chunk_overlap"]
    
    def chunk_document(self, doc_id: str, sections: List[Dict], metadata: Dict) -> List[DocumentChunk]:
        """Chunk a document while preserving section info"""
        chunks = []
        
        for section_data in sections:
            section = section_data["section"]
            text = section_data["text"]
            page_num = section_data.get("page_num")
            
            # Split into sentences first
            sentences = self._split_sentences(text)
            
            # Group sentences into chunks
            current_chunk = []
            current_length = 0
            
            for sent in sentences:
                sent_length = len(sent.split())
                
                if current_length + sent_length > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunk_id = self._generate_chunk_id(doc_id, len(chunks))
                    
                    chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=chunk_text,
                        metadata=metadata,
                        section=section,
                        page_num=page_num
                    ))
                    
                    # Start new chunk with overlap
                    if self.overlap > 0 and len(current_chunk) > 1:
                        # Keep last sentence for overlap
                        current_chunk = [current_chunk[-1], sent]
                        current_length = len(current_chunk[-1].split()) + sent_length
                    else:
                        current_chunk = [sent]
                        current_length = sent_length
                else:
                    current_chunk.append(sent)
                    current_length += sent_length
            
            # Add remaining chunk
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = self._generate_chunk_id(doc_id, len(chunks))
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=chunk_text,
                    metadata=metadata,
                    section=section,
                    page_num=page_num
                ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter"""
        import re
        # Split on . ! ? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_chunk_id(self, doc_id: str, chunk_num: int) -> str:
        """Generate unique chunk ID"""
        return f"{doc_id}_chunk_{chunk_num:04d}"


class EmbeddingEngine:
    """Generate embeddings for text chunks"""
    
    def __init__(self):
        self.model_name = EMBEDDING_CONFIG["text_model"]
        self.model = None
        self.embedding_dim = EMBEDDING_CONFIG["embedding_dim"]
    
    def load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            try:
                from transformers.utils import logging as transformers_logging

                transformers_logging.disable_progress_bar()
            except Exception:
                pass

            try:
                from huggingface_hub.utils import disable_progress_bars

                disable_progress_bars()
            except Exception:
                pass

            # Streamlit and similar browser-backed runtimes can expose a broken
            # stderr pipe during weight loading. Silence stderr while the model
            # initializes so tqdm-based loaders do not crash the app.
            with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
                self.model = SentenceTransformer(self.model_name)
    
    def embed_chunks(self, chunks: List[DocumentChunk], 
                     show_progress: bool = True) -> np.ndarray:
        """Generate embeddings for chunks"""
        self.load_model()
        
        texts = [chunk.text for chunk in chunks]
        show_progress = show_progress and _can_render_live_progress()
        
        if show_progress:
            print(f"Generating embeddings for {len(texts)} chunks...")
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        self.load_model()
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding[0]


class VectorStore:
    """FAISS-based vector store for chunk retrieval"""
    
    def __init__(self):
        self.index = None
        self.chunks = []
        self.chunk_id_to_idx = {}
        self.embedding_dim = EMBEDDING_CONFIG["embedding_dim"]
        self.index_path = VECTOR_STORE_DIR / "index.faiss"
        self.chunks_path = VECTOR_STORE_DIR / "chunks.pkl"
        self.metadata_path = VECTOR_STORE_DIR / "metadata.json"
    
    def build_index(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Build FAISS index from chunks and embeddings"""
        print(f"Building FAISS index for {len(chunks)} chunks...")
        
        # Create index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks
        self.chunks = chunks
        self.chunk_id_to_idx = {
            chunk.chunk_id: idx for idx, chunk in enumerate(chunks)
        }
        
        # Save
        self.save()
        print(f"Index built and saved: {len(chunks)} chunks indexed")
    
    def load(self) -> bool:
        """Load existing index"""
        if not self.index_path.exists():
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            self.chunk_id_to_idx = {
                chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)
            }
            
            print(f"Loaded index: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def save(self):
        """Save index to disk"""
        faiss.write_index(self.index, str(self.index_path))
        
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        metadata = {
            "num_chunks": len(self.chunks),
            "embedding_dim": self.embedding_dim,
            "index_type": str(type(self.index))
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Search for top-k most similar chunks"""
        if self.index is None:
            raise ValueError("Index not loaded. Call load() or build_index() first.")
        
        # Search
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k)
        
        # Return chunks with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                chunk = self.chunks[idx]
                # Convert L2 distance to similarity score (inverse)
                score = 1.0 / (1.0 + dist)
                results.append((chunk, score))
        
        return results


class DataManager:
    """Main interface for data processing pipeline"""
    
    def __init__(self, backend: str = "local"):
        self.pdf_processor = PDFProcessor()
        self.chunker = TextChunker()
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore()
        self.claim_graph = ClaimGraphStore(self.embedding_engine, backend=backend)
        self.figures_db = {}  # doc_id -> List[Figure]
    
    def process_corpus(self, rebuild: bool = False):
        """Process all PDFs in corpus directory"""
        # Check if index exists
        if not rebuild and self.vector_store.load():
            print("Using existing index. Set rebuild=True to reprocess.")
            self._load_figures_db()
            if not self.claim_graph.load():
                self.claim_graph.build_from_chunks(self.vector_store.chunks, rebuild=True)
            return
        
        # Get all PDFs
        pdf_files = list(CORPUS_DIR.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {CORPUS_DIR}")
        
        print(f"Processing {len(pdf_files)} PDFs...")
        
        all_chunks = []
        all_figures = []
        
        pdf_iterator = tqdm(
            pdf_files,
            desc="Processing PDFs",
            disable=not _can_render_live_progress(),
        )
        for pdf_path in pdf_iterator:
            doc_id = self._get_doc_id(pdf_path)
            
            # Extract metadata
            metadata = self.pdf_processor.extract_metadata(pdf_path)
            
            # Extract text with sections
            sections = self.pdf_processor.extract_text_with_sections(pdf_path)
            
            # Chunk document
            chunks = self.chunker.chunk_document(doc_id, sections, metadata)
            all_chunks.extend(chunks)
            
            # Extract figures
            figures = self.pdf_processor.extract_figures(pdf_path, doc_id)
            all_figures.extend(figures)
            self.figures_db[doc_id] = figures
        
        print(f"Total chunks: {len(all_chunks)}")
        print(f"Total figures: {len(all_figures)}")
        
        # Generate embeddings
        embeddings = self.embedding_engine.embed_chunks(all_chunks)
        
        # Build index
        self.vector_store.build_index(all_chunks, embeddings)

        # Build sparse claim graph for fast query-time reasoning
        self.claim_graph.build_from_chunks(all_chunks, rebuild=True)
        
        # Save figures DB
        self._save_figures_db()
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve top-k chunks for a query"""
        query_embedding = self.embedding_engine.embed_query(query)
        return self.vector_store.search(query_embedding, k)
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get chunk by ID"""
        idx = self.vector_store.chunk_id_to_idx.get(chunk_id)
        if idx is not None:
            return self.vector_store.chunks[idx]
        return None

    def retrieve_claim_subgraph(
        self,
        query: str,
        max_claims: Optional[int] = None,
    ):
        """Retrieve a precomputed claim subgraph for fast EVIRAG mode."""
        return self.claim_graph.query_subgraph(query, max_claims=max_claims)
    
    def get_figures_for_doc(self, doc_id: str) -> List[Figure]:
        """Get all figures for a document"""
        return self.figures_db.get(doc_id, [])
    
    def _get_doc_id(self, pdf_path: Path) -> str:
        """Generate document ID from path"""
        # Use hash of filename for uniqueness
        return hashlib.md5(pdf_path.name.encode()).hexdigest()[:12]
    
    def _save_figures_db(self):
        """Save figures database"""
        figures_db_path = CACHE_DIR / "figures_db.pkl"
        with open(figures_db_path, 'wb') as f:
            pickle.dump(self.figures_db, f)
    
    def _load_figures_db(self):
        """Load figures database"""
        figures_db_path = CACHE_DIR / "figures_db.pkl"
        if figures_db_path.exists():
            with open(figures_db_path, 'rb') as f:
                self.figures_db = pickle.load(f)


if __name__ == "__main__":
    # Test the data layer
    manager = DataManager()
    manager.process_corpus(rebuild=False)
    
    # Test query
    test_query = "How does attention mechanism work in transformers?"
    results = manager.retrieve(test_query, k=5)
    
    print(f"\nQuery: {test_query}")
    print(f"Top {len(results)} results:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.3f}")
        print(f"   Document: {chunk.metadata.get('title', 'Unknown')}")
        print(f"   Section: {chunk.section}")
        print(f"   Text: {chunk.text[:200]}...")
