# Enterprise RAG Pipeline: High-Speed Document Intelligence

Let's be honest, building a RAG (Retrieval-Augmented Generation) system sounds easy until you actually try to feed it a 600-page scanned textbook. That is usually when the memory fills up, the text extraction stalls, and your neat two-second latency target turns into a 15-minute nightmare. 

This project is my answer to that exact problem. I built a highly optimised, locally-indexable RAG pipeline designed to chew through massive corporate PDFs, construct a fast vector search graph, and deliver cited answers in under five seconds.

## What It Actually Does

At its core, this system lets you upload hundreds of pages of PDF documents, embeds the text into a vector database, and allows you to chat with your data using Google's Gemini API. 

However, the real magic happens under the hood. I didn't just plug in LangChain and call it a day. I had to heavily optimise the ingestion layer to survive strict hardware limits (like standard dual-core CPUs).

### Key Features
* **Hybrid Search Retrieval:** Relying purely on basic vector distance isn't enough. This pipeline uses a Bi-Encoder (`BAAI/bge-small-en-v1.5`) to surface the top 15 candidate chunks, and a Cross-Encoder (`BAAI/bge-reranker-base`) to score and filter the top 4. This practically eliminates those frustrating 'lost-in-the-middle' LLM hallucinations.
* **Layout-Aware Ingestion:** The text extractor uses PyMuPDF for lightning-fast native digital text extraction, seamlessly falling back to Tesseract OCR when it detects a scanned layout.
* **HNSW Graph Optimisation:** I tuned the Qdrant vector database parameters (`m=16`, `ef_construct=64`) to force sub-150ms search times.
* **Sub-5 Second Latency:** Even with the heavy Cross-Encoder reranking, the total time from the user hitting 'Enter' to the LLM streaming the answer stays comfortably within the 2 to 5-second window.

## The Tech Stack

I kept the stack entirely open-source for the vectorisation and retrieval, only offloading the final generative step to a hosted API to save local VRAM and guarantee low latency.

* **Parsing & OCR:** PyMuPDF (Fitz), Tesseract OCR, Pillow
* **Embeddings & Reranking:** SentenceTransformers (PyTorch)
* **Vector Database:** Qdrant (Local persistent storage)
* **Generation:** Google Gemini 2.5 Flash
* **Frontend:** Streamlit

## Overcoming The Ingestion Bottleneck

During development, I hit a massive wall: a 690-page scanned PDF. Standard OCR on a basic CPU takes about three to four seconds per page. Do the maths, and that is nearly 45 minutes of processing time—an absolute death sentence for a time-capped demo.

**Here is how I solved it:**
1. **Multiprocessing:** I wrapped the OCR extraction in Python's `ProcessPoolExecutor` to force the system to utilise all available CPU cores concurrently.
2. **Matrix Downscaling:** I fed Tesseract a downscaled matrix (72 DPI) of the PDF pages. It was still perfectly legible for the AI to extract the words, but it cut the image byte size by 80%, dropping processing time to under a second per page.
3. **GPU Offloading:** I explicitly mapped the SentenceTransformer models to target the CUDA/T4 GPU, reducing the embedding generation time for thousands of chunks down to mere seconds.

## Running It Locally

If you want to spin this up on your own machine, you will need a decent GPU if you plan on ingesting hundreds of pages quickly.

### Prerequisites
Make sure you have Tesseract installed on your system (`sudo apt install tesseract-ocr` on Linux, or via Homebrew on Mac).

```bash
# Clone the repository
git clone [https://github.com/CodeAlchemist09/RAG-Chatbot.git](https://github.com/CodeAlchemist09/RAG-Chatbot.git)
cd RAG-Chatbot

# Set up a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
