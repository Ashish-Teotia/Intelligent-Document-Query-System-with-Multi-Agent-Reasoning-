# Intelligent-Document-Query-System-with-Multi-Agent-Reasoning-


# Project Title

A brief description of what this project does and who it's for
# Multi-Agent RAG System

This project implements a multi-agent Retrieval-Augmented Generation (RAG) pipeline using Azure OpenAI, LangChain, and web search APIs. It processes DOCX documents, extracts and explains tables/images, builds a vector database, and answers user queries using both document context and web search. An evaluator agent selects the best answer.

## Features

- **Document Processing:** Loads DOCX files, extracts text, tables, and images, and uses LLMs to explain tables/images.
- **Chunking & Embedding:** Splits documents into chunks and embeds them using SentenceTransformer.
- **Vector Database:** Stores chunk embeddings in a Chroma vector DB for similarity search.
- **RAG Agent:** Answers queries using only document context.
- **Web Search Agent:** Answers queries using web search snippets.
- **Evaluator Agent:** Compares both answers and selects the best one.

## File Structure

- [`main.py`](main.py): Entry point; orchestrates the workflow.
- [`docs_utility.py`](docs_utility.py): DOCX loading, table/image explanation, chunking.
- [`vector_utils.py`](vector_utils.py): Embedding, vector DB creation, similarity filtering.
- [`agent1.py`](agent1.py): RAG agent logic.
- [`agent_web.py`](agent_web.py): Web search agent logic.
- [`evaluator_agent.py`](evaluator_agent.py): Evaluator agent logic.
- `azure_config_list.py`: Azure OpenAI configuration (not included).

## Setup
1. **Configure Azure OpenAI:**
    - Create `azure_config_list.py` with your Azure OpenAI credentials and deployment details.

2. **Prepare Documents:**
    - Place DOCX files in the folder specified in `main.py` (default: `C:\Users\ashish.i.choudhary\Latest Code\rag\documents`).

## Usage

Run the main script:

```sh
python main.py
```

- Enter your query when prompted.
- The system will process documents, run both agents, and print the best answer.

## Notes

- Requires access to Azure OpenAI and Search1API.
- Make sure your API keys and endpoints are correct.
- The project is designed for experimentation and can be extended for other document types or agents.


