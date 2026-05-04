# 🚑 MedRoute — Adaptive Hybrid KG–RAG Medical Assistant

An AI-powered healthcare query assistant that combines **Knowledge Graphs, Hybrid Retrieval, and Adaptive Routing** to deliver **accurate, context-aware, and explainable medical responses**.

---

## 🏗️ System Architecture

```mermaid
graph TD
    U[User / Clinician] --> F[Streamlit Frontend]
    F --> B[FastAPI Backend]

    B --> R[Adaptive Query Router]

    R --> G[Graph Retrieval\n(Cypher - Neo4j)]
    R --> K[BM25 Keyword Search]
    R --> V[Vector Retrieval\n(Embeddings)]
    R --> L[Direct LLM Response]

    G --> M[RRF Fusion Layer]
    K --> M
    V --> M

    M --> A[LLM (Ollama)]
    L --> A

    A --> O[Final Response + Validation Layer]
## 🚀 Key Features

### 🧠 Hybrid Retrieval System
- Graph-based retrieval (Neo4j Cypher queries)
- BM25 keyword search
- Vector similarity search

### 🔀 Adaptive Query Routing
- Classifies queries into:
  - Factual  
  - Relational  
  - Complex  
  - General  
- Dynamically selects optimal retrieval strategy

### 🔗 Knowledge Graph Integration
- Models relationships between diseases, symptoms, and treatments  
- Enables multi-hop reasoning  

### ⚡ Reciprocal Rank Fusion (RRF)
- Combines multiple retrieval results  
- Improves ranking accuracy and relevance  

### 🤖 Local LLM (Ollama Integration)
- Uses LLaMA-based models for response generation  
- Ensures privacy and offline capability  

### 🛡️ Low Hallucination Output
- RAG grounding ensures factual consistency  
- Observed **0% hallucination** in evaluation  

### 📊 Evaluation Pipeline
- 60-query benchmark testing  
- Measures:
  - Accuracy  
  - Relevancy  
  - Coverage  
  - Routing Precision  

---

## 📌 Problem Statement

Traditional healthcare search systems:

- ❌ Fail to understand user intent  
- ❌ Cannot handle relational medical queries  
- ❌ Provide unverified or generic responses  
- ❌ Lack contextual reasoning  

---

## 💡 Proposed Solution

**MedRoute** introduces a hybrid architecture that combines:

- Knowledge Graph reasoning  
- Multi-channel retrieval (Graph + BM25 + Vector)  
- Adaptive query routing  
- Retrieval-Augmented Generation (RAG)  

👉 Result: **Accurate, explainable, and context-aware healthcare responses**

---

## 📊 Performance Highlights

| Metric | Value |
|--------|------|
| Accuracy | **68.7%** |
| Relevancy | **71.8%** |
| Coverage | **74.7%** |
| Routing Precision | **100%** |
| Hallucination Rate | **0%** |

---

### 🔍 Key Observations

- Adaptive routing improves completeness (**4.08 vs 3.74**)  
- Complex queries remain most challenging  
- Hybrid retrieval significantly boosts relevance  

---

## 🛠️ Tech Stack

### Backend
- Python  
- FastAPI  

### Frontend
- Streamlit  

### AI / ML
- LangChain  
- Ollama (LLaMA 3 / local LLM)  
- RAG Pipeline  

### Database
- Neo4j Graph Database  
- Vector Embeddings (Neo4j / local)

### Retrieval
- BM25  
- Dense Vector Search  
- Cypher Query Engine  

📂 Project Structure
MED-ROUTE-final/
│
├── chatbot_api/
│   ├── src/
│   │   ├── agents/
│   │   ├── ingest/
│   │   ├── retrieval/
│   │   └── main.py
│
├── chatbot_frontend/
│   └── src/
│
├── tests/
│   ├── eval/
│   └── run_route_eval_60.py
│
└── README.md

1. Clone the repository
git clone https://github.com/PratikSangde201/MedRoute-.git
cd MedRoute-
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
4. Run Backend
cd chatbot_api/src
python main.py
5. Run Frontend
cd chatbot_frontend/src
python main.py

🧪 Evaluation
python tests/run_route_eval_60.py
Used to measure routing accuracy and response performance.

🔄 Workflow
User → Frontend → Backend → Router → RAG → Response
🔮 Future Enhancements
Real-time hospital/emergency integration
Location-based intelligent routing
Voice-enabled assistant
Mobile application support



