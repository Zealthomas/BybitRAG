from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / os.getenv("PERSIST_DIR", "faiss_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

embedding_model: Optional[HuggingFaceEmbeddings] = None
llm: Optional[OllamaLLM] = None
retriever = None
vector_db = None

DEFAULT_URLS = [
    "https://bybit-exchange.github.io/docs",
    "https://www.bybit.com/en/help-center/",
    "https://www.bybit.com/en/announcement/",
    "https://www.bybit.com/en/learn/",
    "https://www.bybit.com/en-US/legal/terms-of-service",
    "https://www.bybit.com/en/fees/",
    "https://blog.bybit.com/",
    "https://www.bybit.com/en-US/affiliate",
]


def ensure_persist_dir() -> None:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)


def init_models() -> None:
    global embedding_model, llm

    if embedding_model is None:
        try:
            embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            logging.info("Embedding model initialized.")
        except Exception as exc:
            logging.warning("Embedding model initialization failed: %s", exc)
            embedding_model = None

    if llm is None:
        try:
            llm = OllamaLLM(model=OLLAMA_MODEL)
            logging.info("Ollama model initialized.")
        except Exception as exc:
            logging.warning("Ollama model initialization failed: %s", exc)
            llm = None


def load_vector_db() -> Optional[FAISS]:
    global vector_db
    if not PERSIST_DIR.exists() or not any(PERSIST_DIR.iterdir()):
        return None
    if embedding_model is None:
        return None
    try:
        db = FAISS.load_local(str(PERSIST_DIR), embedding_model, allow_dangerous_deserialization=True)
        vector_db = db
        return db
    except Exception as exc:
        logging.warning("Failed to load FAISS index; rebuilding: %s", exc)
        if PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
        return None


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    selectors = [
        "article",
        "main",
        ".article-body",
        ".content",
        ".entry-content",
        "#content",
        "#main-content",
    ]
    main_content = None
    for selector in selectors:
        main_content = soup.select_one(selector)
        if main_content is not None:
            break
    if main_content is None:
        main_content = soup.body or soup

    for tag in main_content.find_all(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    text = main_content.get_text(separator="\n", strip=True)
    return " ".join(text.split())


def fetch_and_extract_text(url: str) -> str:
    headers = {"User-Agent": "BybitRAGBot/1.0"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    text = extract_main_text(response.text)
    if not text or len(text) < 100:
        raise ValueError(f"No usable content found at {url}")
    return text


def create_documents_from_urls(urls: List[str]) -> List[Document]:
    documents: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for url in urls:
        try:
            content = fetch_and_extract_text(url)
            chunks = splitter.split_text(content)
            for chunk in chunks:
                documents.append(Document(page_content=chunk, metadata={"source": url}))
            logging.info("Indexed %s chunks from %s", len(chunks), url)
        except Exception as exc:
            logging.warning("Failed to index %s: %s", url, exc)

    if not documents:
        raise ValueError("No documents were successfully extracted from the provided URLs")

    return documents


def ingest_documents(urls: List[str]) -> FAISS:
    global vector_db
    if embedding_model is None:
        raise RuntimeError("Embedding model is not initialized")

    docs = create_documents_from_urls(urls)
    db = FAISS.from_documents(docs, embedding_model)
    ensure_persist_dir()
    db.save_local(str(PERSIST_DIR))
    vector_db = db
    logging.info("Vector store saved to %s", PERSIST_DIR)
    return db


def ensure_retriever() -> None:
    global retriever
    if vector_db is not None:
        retriever = vector_db.as_retriever(search_kwargs={"k": 4})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db, retriever
    init_models()

    if embedding_model is not None:
        db = load_vector_db()
        if db is None:
            try:
                db = ingest_documents(DEFAULT_URLS)
            except Exception as exc:
                logging.warning("Failed to build initial database: %s", exc)
                db = None
        vector_db = db
        if db is not None:
            ensure_retriever()

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embedding_ready": embedding_model is not None,
        "llm_ready": llm is not None,
        "vector_db_ready": vector_db is not None,
        "persist_dir": str(PERSIST_DIR),
    }


@app.post("/ingest")
async def ingest_endpoint(request: Request):
    global vector_db, retriever
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    except Exception:
        body = {}

    url = body.get("url") or request.query_params.get("url")
    if not url:
        return JSONResponse(status_code=400, content={"error": "Missing URL"})

    if embedding_model is None:
        return JSONResponse(status_code=503, content={"error": "Embedding model is unavailable"})

    try:
        docs = create_documents_from_urls([url])
        if vector_db is None:
            vector_db = FAISS.from_documents(docs, embedding_model)
        else:
            vector_db.add_documents(docs)
        ensure_persist_dir()
        vector_db.save_local(str(PERSIST_DIR))
        ensure_retriever()
        return {"status": "success", "message": f"Ingested content from {url}"}
    except Exception as exc:
        logging.exception("Manual ingest failed")
        return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    query = (body or {}).get("query") or request.query_params.get("query") or ""
    if not query or not query.strip():
        return JSONResponse(status_code=400, content={"error": "Query is required"})

    if llm is None or retriever is None:
        return JSONResponse(status_code=503, content={"error": "LLM or retriever is not ready"})

    try:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
        result = qa_chain.invoke({"query": query})
        answer = result.get("result") if isinstance(result, dict) else str(result)
        sources = []
        if isinstance(result, dict):
            for doc in result.get("source_documents", []):
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "snippet": doc.page_content[:200],
                })
        return {"response": answer, "sources": sources}
    except Exception as exc:
        logging.exception("Chat request failed")
        return JSONResponse(status_code=500, content={"detail": str(exc)})


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)
