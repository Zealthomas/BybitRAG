# ========== IMPORTS ==========
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os
import requests
from bs4 import BeautifulSoup

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# ========== GLOBALS ==========
persist_dir = "faiss_db"

embedding_model = None
llm = None
retriever = None

# ========== HELPERS ==========
def fetch_and_extract_text(url):
    print(f"Fetching URL: {url}")
    headers = {"User-Agent": "BybitRAGBot/1.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    main_content = soup.find("div", {"class": "article-body"}) or soup.body

    for tag in main_content(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = main_content.get_text(separator="\n", strip=True)
    print(f"Extracted {len(text)} characters of text from {url}")
    return text

def ingest_documents(urls):
    print("Starting document ingestion...")
    texts = []
    metadatas = []

    for url in urls:
        try:
            text = fetch_and_extract_text(url)
            if text.strip():
                texts.append(text)
                metadatas.append({"source": url})
            else:
                print(f"Warning: No extractable text from {url}")
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for i, text in enumerate(texts):
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=metadatas[i]))

    print(f"Total chunks created: {len(docs)}")

    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(persist_dir)
    print(f"FAISS index saved to {persist_dir}")
    return db

# ========== LIFESPAN HANDLER ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, llm, retriever

    if embedding_model is None:
        print("Initializing embedding model and LLM...")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        llm = OllamaLLM(model="mistral")
        print("Embedding model and LLM initialized.")

    if not os.path.exists(persist_dir):
        print("No existing FAISS index found, ingesting documents...")
        db = ingest_documents([
            "https://bybit-exchange.github.io/docs",
            "https://www.bybit.com/en/help-center/",
            "https://www.bybit.com/en/announcement/",
            "https://www.bybit.com/en/learn/",
            "https://www.bybit.com/en-US/legal/terms-of-service",
            "https://www.bybit.com/en/fees/",
            "https://blog.bybit.com/",
            "https://www.bybit.com/en-US/affiliate"
        ])
    else:
        print(f"Loading existing FAISS index from {persist_dir}...")
        db = FAISS.load_local(persist_dir, embedding_model, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")

    retriever = db.as_retriever()
    print("Retriever is ready.")

    yield
    # (You can add shutdown code here if needed)

# ========== APP SETUP ==========
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== INGEST ENDPOINT ==========
@app.post("/ingest")
async def ingest_endpoint(request: Request):
    global retriever
    try:
        url = request.query_params.get("url")
        print(f"Received /ingest request with URL: {url}")
        if not url:
            return JSONResponse(status_code=400, content={"error": "Missing URL"})

        db = ingest_documents([url])
        retriever = db.as_retriever()
        print("Manual ingestion complete.")
        return {"status": "success", "message": f"Ingested content from {url}"}

    except Exception as e:
        print(f"Error in /ingest endpoint: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ========== CHAT ENDPOINT ==========
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        print(f"/chat received data: {data}")
        query = data.get("query", "").strip()

        if not query:
            print("Empty query received.")
            return JSONResponse(status_code=400, content={"error": "Query is required"})

        if retriever is None:
            print("Retriever not initialized.")
            return JSONResponse(status_code=500, content={"error": "Retriever not initialized"})

        print("Running retrieval QA chain...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        response = qa_chain.run(query)
        print(f"Query response: {response}")
        return {"response": response}

    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

# ========== RUN APP LOCALLY ==========
if __name__ == "__main__":
    print("Starting FastAPI app...")
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)

# ========== END OF FILE ==========
