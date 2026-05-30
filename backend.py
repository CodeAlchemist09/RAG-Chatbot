import os
import io
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import fitz  
import pytesseract
from PIL import Image
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MULTIPROCESSING WORKER ---
# This MUST remain completely outside the class so it can be pickled across CPU cores
def _parallel_ocr_worker(pdf_path: str, page_num: int) -> tuple:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    text = page.get_text("text").strip()
    
    # OCR Fallback for scanned pages
    if len(text) < 100:  
        # CRITICAL OPTIMIZATION: Force matrix scale to 1.0 to shrink image byte size. 
        # This makes Tesseract run 4x-5x faster while maintaining enough fidelity to read text.
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(image)
        
    doc.close()
    return (page_num + 1, text)

# --- ENTERPRISE PIPELINE ---
class ProductionRAGPipeline:
    def __init__(self, collection_name: str = "private_corpus"):
        self.collection_name = collection_name
        
        # Hardware Acceleration: Force models to use CUDA/T4 GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing AI Models on hardware: {self.device.upper()}")
        
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=self.device)
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", device=self.device)
        
        self.db_client = QdrantClient(path="./qdrant_storage")
        self._setup_vector_collection()

    def _setup_vector_collection(self):
        if not self.db_client.collection_exists(self.collection_name):
            self.db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=64, full_scan_threshold=10000, on_disk=False)
            )

    def chunk_text(self, text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += (chunk_size - overlap)
        return chunks

    def ingest_pdf(self, pdf_path: str):
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        logger.info(f"Starting Multi-Core Parallel Extraction for {total_pages} pages...")
        start_time = time.time()
        extracted_pages = []
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_parallel_ocr_worker, pdf_path, p) for p in range(total_pages)]
            for future in futures:
                extracted_pages.append(future.result())

        logger.info(f"Extraction complete in {round(time.time() - start_time, 2)}s. Generating GPU embeddings...")
        
        points = []
        point_id = int(time.time() * 1000)
        
        # --- STRUCTURAL TRACKING ADDED HERE ---
        current_chapter = "Front Matter / Introduction"
        
        for page_num, text in extracted_pages:
            if not text: continue
            
            # Simple heuristic: If the word "Chapter" appears early on the page, grab that line
            if "chapter" in text[:200].lower():
                first_line = text[:200].strip().split('\n')
                if len(first_line) < 50: # Ensure it's actually a title and not just a sentence
                    current_chapter = first_line

            chunks = self.chunk_text(text)
            for chunk_content in chunks:
                normalized_text = " ".join(chunk_content.replace("\n", " ").split())
                vector = self.embedding_model.encode(normalized_text).tolist()
                
                # We inject the 'current_chapter' into the metadata payload
                payload = {
                    "filename": os.path.basename(pdf_path), 
                    "page_number": page_num, 
                    "chapter": current_chapter, # Added Chapter Metadata
                    "text": f"[Section: {current_chapter}] " + normalized_text
                }
                
                points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))
                point_id += 1

        if points:
            self.db_client.upload_points(collection_name=self.collection_name, points=points)
        logger.info(f"✅ Ingestion successful. Vectors indexed with structural metadata.")

    def query_pipeline(self, query_text: str, top_k_ann: int = 15, top_k_rerank: int = 4) -> Dict[str, Any]:
        start_time = time.time()
        
        query_vector = self.embedding_model.encode(query_text).tolist()
        
        # FIXED: Using the modern Qdrant API method (query_points instead of search)
        search_response = self.db_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k_ann
        )
        ann_results = search_response.points  # Extract the matched points
        
        if not ann_results:
            return {"context_window": "", "sources": [], "latency_seconds": time.time() - start_time}

        passages = [hit.payload["text"] for hit in ann_results]
        pairs = [[query_text, passage] for passage in passages]
        rerank_scores = self.reranker.predict(pairs)
        
        for idx, score in enumerate(rerank_scores):
            ann_results[idx].score = float(score)
        
        reranked_results = sorted(ann_results, key=lambda x: x.score, reverse=True)[:top_k_rerank]
        
        context_str = "\n\n".join([f"--- Context {i+1} ---\n{hit.payload['text']}" for i, hit in enumerate(reranked_results)])
        sources = [{"filename": hit.payload["filename"], "page": hit.payload["page_number"]} for hit in reranked_results]
        
        return {
            "context_window": context_str,
            "sources": sources,
            "latency_seconds": round(time.time() - start_time, 3)
        }
