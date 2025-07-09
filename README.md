# 🟡 BybitRAG: Retrieval-Augmented AI Assistant for Bybit Users

> ⚡️ A sleek, interactive frontend prototype for an intelligent AI assistant built to support Bybit traders, developers, and users. This is part of a larger **RAG-based system** powered by open-source LLMs (Mistral 7B, Gemma) and custom backend logic — currently under development.

---

## 📸 Project Purpose & Demo Use

This project is part of a multi-phase showcase for a production-grade Retrieval-Augmented Generation (RAG) tool. It is intended to:

- ✅ Demonstrate frontend UI/UX for Bybit Assistant  
- ✅ Share development progress publicly  
- 🔄 Evolve into a full-stack LLM RAG application  
- 🚀 Attract collaboration and feedback from the dev community  

> The code here is real, but the backend is **mocked** for now to simulate AI responses while the API and vector DB integration are being built.

---

## 🎯 Features (Phase 1: Frontend Prototype)

- 🎨 Glassmorphism UI with dark theme + Bybit color palette  
- 💬 Interactive chat layout (user vs assistant)  
- 🤖 Mock streaming responses using JavaScript  
- 🔍 Knowledge base and chat history panes  
- 📈 Stats & typing animations  
- 🧠 Sample questions triggering preset responses  
- ♿ Fully responsive on mobile & desktop  

---

## 📁 File Structure

```plaintext
BybitRAG-Frontend/
├── index.html       # Complete frontend UI (HTML, CSS, JS inlined)
├── .git/            # Git metadata
└── README.md        # This file


🛠 Local Preview (No Backend Needed)
Clone the repo:
git clone https://github.com/Zealthomas/BybitRAG-Frontend.git
cd BybitRAG-Frontend
Open index.html in your browser:

Double-click it

Or right-click → Open with → Chrome/Edge/Firefox

📦 Planned Roadmap
🔄 In Progress
Backend: FastAPI with RAG pipeline

Embeddings: SentenceTransformers + ChromaDB

Text Generation: Mistral 7B via Ollama
Data: Custom Bybit documentation dataset

✅ Done
Complete responsive frontend

Mock assistant flow with referencs for demos

Sidebar logic for KB + history

Typing indicators and UI polish

Live demo will be visible at:
https://zealthomas.github.io/BybitRAG-Frontend/
🤝 Author & Dev
👨‍💻 N. Thomas

GitHub: @Zealthomas

Twitter: @Zealthomas

LinkedIn: in/nkpoikankethomas

Project: Build and deploy an open-source, production-grade RAG system with real-time reasoning, targeting crypto platforms like Bybit and Binance.

📄 License
This project is licensed under the MIT License.
