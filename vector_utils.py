# vector_utils.py

import os
import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Initialize embedding model globally
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def create_vector_db_from_chunks(chunks, persist_directory="./chroma_db"):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"Vector DB already exists at: {persist_directory}. Skipping embedding.")
        return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    print("Creating new vector DB from chunks...")
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector DB created and persisted at: {persist_directory}")
    return vectordb


def embed_query_vector(query):
    """
    Return the embedding vector for a given query string.
    """
    embedding = embedding_model.embed_query(query)
    print(f"Query embedding generated for: '{query}'")
    return embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def filter_chunks_by_similarity(query_embedding, threshold=0.5, persist_directory="./chroma_db"):
    """
    Loads vector store, calculates similarity of each chunk to the query,
    and returns filtered chunks with similarity > threshold.
    """
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    data = vectorstore.get(include=["documents", "embeddings", "metadatas"])

    docs = data['documents']
    embeddings = data['embeddings']
    existing_metadatas = data['metadatas']

    filtered_docs = []
    new_metadatas = []

    for i in range(len(docs)):
        sim = cosine_similarity(query_embedding, embeddings[i])
        if sim > threshold:
            meta = existing_metadatas[i] if existing_metadatas[i] else {}
            meta["similarity_to_query"] = sim
            filtered_docs.append(docs[i])
            new_metadatas.append(meta)

    # Combine into a displayable structure
    chunk_similarity_data = [
        {"chunk": doc, "similarity": meta["similarity_to_query"]}
        for doc, meta in zip(filtered_docs, new_metadatas)
    ]

    return chunk_similarity_data
