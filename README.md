# mutual_funds_recommendation_system

An AI-powered recommendation engine that leverages **Retrieval-Augmented Generation (RAG)**, semantic search, and financial data to provide personalized, explainable mutual fund suggestions.

---

This project introduces a **Mutual Funds Recommendation System** that combines:
- **FAISS-based semantic search** for fund retrieval
- **Real-time NAV data** via external APIs
- **Phi-2 LLM fine-tuned with LoRA** for query rewriting and explanation generation
- **Hybrid retrieval (semantic + keyword filtering)** for context-aware fund selection
- **Weighted scoring algorithm** based on NAV trends, historical returns, and expense ratios

The system delivers **transparent, goal-aligned recommendations** with human-readable explanations, making investment decisions easier for retail investors.

---

## 🚀 Features
- **Natural Language Queries**: Users can ask in plain English (e.g., “Suggest safe funds for 5 years”).  
- **Semantic Retrieval**: FAISS + MiniLM embeddings for fund similarity search.  
- **Real-Time NAV Integration**: Fetches live NAV values via APIs.  
- **LLM-Powered Explanations**: Fine-tuned Phi-2 generates clear financial reasoning.  
- **Fund Ranking**: Weighted scoring based on NAV patterns, returns, and expense ratio.  
- **Visualization Dashboard**: Charts comparing NAV and returns.  
- **Frontend-Backend Integration**: React frontend + FastAPI backend.  

---

## 🛠️ Tech Stack
- **Languages**: Python, JavaScript (React)  
- **Backend**: FastAPI, Ngrok (for tunneling)  
- **Frontend**: React.js  
- **Libraries**:  
  - `pandas`, `numpy` → data preprocessing  
  - `faiss-cpu`, `sentence-transformers` → semantic search  
  - `transformers`, `peft`, `accelerate` → LLM fine-tuning (Phi-2 with LoRA)  
  - `matplotlib` → chart visualization  
  - `evaluate` → BLEU score evaluation  

