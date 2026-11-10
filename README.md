Title:RAG Evaluation & Hallucination Detection Framework

A Python-based framework for evaluating Retrieval-Augmented Generation (RAG) systems through document retrieval experiments, evidence-support analysis, unsupported-claim detection, and performance monitoring.

1.Overview

Retrieval-Augmented Generation systems can produce answers that are not fully supported by retrieved documents. This project provides an experimental framework to study retrieval configurations and measure how well generated answers are supported by retrieved evidence.

It uses a locally hosted Large Language Model through Ollama, ChromaDB for vector storage, Sentence Transformers for embeddings, and a Streamlit dashboard for experiment analysis.

2.Key Features

- PDF Processing: Extracts text from PDF documents, cleans content, and splits it into manageable chunks.
- Vector Search: Converts document chunks into embeddings and stores them in ChromaDB.
- RAG Pipeline: Retrieves relevant document sections and generates answers using a local LLM.
- Evidence-Support Evaluation: Checks generated statements against retrieved evidence using keyword coverage and basic negation detection.
- Unsupported-Claim Detection: Identifies statements that do not meet the configured evidence-support threshold.
- Retrieval Experiments: Compares different chunk sizes and retrieval configurations.
- Performance Tracking: Records evidence-support percentages and query execution times.
- Interactive Dashboard: Visualizes experiment results using Streamlit and Plotly.
- Automated Testing: Includes unit tests and ChromaDB integration tests using Pytest.

3.Architecture

             PDF Documents
                   |
                   v
           Document Processor
          (Extraction & Chunking)
                   |
                   v
          Sentence Transformers
               Embeddings
                   |
                   v
              ChromaDB
             Vector Store
                   |
             User Question
                   |
                   v
             RAG Pipeline
                   |
             Retrieve Context
                   |
                   v
              Ollama LLM
             (llama3.2:3b)
                   |
             Generated Answer
                   |
                   v
          Evidence-Support Evaluator
                   |
          Supported / Unsupported
               Statements
                   |
                   v
          Experiment Results (CSV)
                   |
                   v
          Streamlit Dashboard


4.Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| LLM | Ollama – Llama 3.2 (3B) |
| Embeddings | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| PDF Processing | PyPDF |
| Data Processing | Pandas, NumPy |
| Dashboard | Streamlit, Plotly |
| Testing | Pytest |
| Version Control | Git, GitHub |

5.Project Structure

RAG_Evaluation_Framework/
│
├── app.py
├── requirements.txt
├── requirements-lock.txt
│
├── src/
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── evaluator.py
│   ├── hallucination_detector.py
│   └── experiment_runner.py
│
├── data/
│   ├── documents/
│   │   └── sample.pdf
│   └── evaluation/
│
├── reports/
│   └── experiment_results.csv
│
├── tests/
│   ├── test_document_processor.py
│   ├── test_evaluator.py
│   ├── test_evaluator_unit.py
│   ├── test_hallucination_cases.py
│   ├── test_hallucination_detector.py
│   ├── test_ragas_prompt.py
│   ├── test_rag_pipeline.py
│   ├── test_vector_store.py
│   └── test_vector_store_integration.py
│
└── .gitignore


6.How It Works

1. Upload or provide a PDF document.
2. Extract and clean the document text.
3. Split the extracted content into smaller chunks.
4. Generate embeddings using Sentence Transformers.
5. Store the embeddings in ChromaDB.
6. Retrieve relevant chunks for a user question.
7. Generate an answer using the local Ollama model.
8. Extract factual statements from the generated answer.
9. Compare statements with retrieved evidence.
10. Calculate evidence-support scores and save experiment results.
11. Analyze the results through the Streamlit dashboard.

7.Experimental Evaluation

Two retrieval configurations were tested using the sample PDF and five questions per configuration.

| Configuration | Chunk Size | Overlap | Top K |
|---|---:|---:|---:|
| Experiment A | 500 | 100 | 2 |
| Experiment B | 800 | 120 | 2 |

8.Observed Results

| Metric | Experiment A | Experiment B |
|---|---:|---:|
| Total Chunks | 4 | 2 |
| Average Evidence Support | 95.00% | 100.00% |
| Average Query Time | 4.50 sec | 4.02 sec |

Across the ten recorded question results:

- Average evidence support: 97.50%
- Average query execution time: 4.26 seconds
- Supported statements: 22
- Unsupported statements: 1

These are observations from a small experiment using one sample PDF and five questions per configuration. They are not general performance guarantees.

9.Evaluation Method

The current evaluator uses:

- Statement extraction from generated answers.
- Keyword coverage comparison against retrieved evidence.
- A configurable evidence-support threshold.
- Basic negation detection to identify some contradictory statements.
- Supported and unsupported statement counts.
- An overall evidence-support percentage.

Important limitation:This is a lightweight heuristic evaluation method, not a validated semantic entailment model or a successful RAGAS faithfulness evaluation. It may miss paraphrases, subtle contradictions, and complex factual relationships. The reported evidence-support percentage should not be interpreted as a definitive hallucination rate.

10.Installation and Setup

1. Clone the repository

git clone: https://github.com/valli0796/rag-evaluation-hallucination-detection.git
cd RAG_Evaluation_Framework


2. Create a virtual environment

python -m venv venv

Activate it on Windows:venv\Scripts\activate

3. Install dependencies

python -m pip install -r requirements.txt

4. Install and start Ollama

Install Ollama and download the required model:ollama pull llama3.2:3b
Ensure the Ollama service is running locally.

5. Run the experiment:python -m src.experiment_runner

6. Launch the dashboard:streamlit run app.py


11.Testing

Run the test suite:python -m pytest -v
The tests cover document processing, vector storage, RAG pipeline behavior, evidence evaluation, hallucination-detection cases, and integration with ChromaDB.

12.Future Improvements

- Integrate a reliable semantic entailment model for evidence verification.
- Add a validated RAGAS evaluation pipeline.
- Evaluate retrieval precision and recall using labeled reference datasets.
- Support multiple embedding models and LLMs.
- Add larger benchmark datasets and repeated experiments.
- Introduce automated regression testing for retrieval quality.
- Track experiment history and compare model configurations.

13.Project Objective

The objective is to build a reproducible evaluation framework that helps developers investigate RAG retrieval quality, analyze generated answers against retrieved evidence, and monitor system behavior across different configurations.