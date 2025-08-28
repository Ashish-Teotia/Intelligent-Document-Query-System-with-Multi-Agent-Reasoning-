# main.py
import numpy as np
import asyncio
from docs_utility import load_docx_documents, split_text_into_chunks
from vector_utils import (
    create_vector_db_from_chunks,
    embed_query_vector,
    filter_chunks_by_similarity
)

from agent1 import agent_rag, run_agent_with_retry  # import from new agent1.py
from agent_web import run_web_agent_with_retry
from agent_web import web_agent
from agent_web import web_search
from evaluator_agent import run_evaluator_agent
import azure_config_list # your config with model_details dict

async def main():
    folder_path=r'C:\Users\ashish.i.choudhary\Latest Code\rag\documents'
    model_data = azure_config_list.model_details

    # Step 1: Load PDFs
    documents, doc_paths = load_docx_documents(folder_path,model_data)

    # Step 2: Split the first document into chunks
    print("\n--- Splitting first document into chunks ---")
    chunks = split_text_into_chunks(documents[0])

    # Step 3: Create and persist vector database (if not already created)
    print("\n--- Creating and persisting vector database ---")
    create_vector_db_from_chunks(chunks)

    # Step 4: Embed the query
    print("Enter the query?")
    query = input()
    print(f"\n--- Generating embedding for query: '{query}' ---")
    query_embedding = embed_query_vector(query)
    query_embedding = np.array(query_embedding)  # Convert to numpy array


    # Step 5: Filter relevant chunks using cosine similarity
    print("\n--- Filtering similar chunks based on cosine similarity ---")
    similar_chunks = filter_chunks_by_similarity(query_embedding, threshold=0.5)

    # Step 6: Print top 3 results
    for item in similar_chunks[:3]:
        print(f"Similarity: {item['similarity']:.4f}")
        print(f"Chunk: {item['chunk'][:150]}...\n")

    # Step 7: Build RAG context
    rag_context = ".".join(f"- {item['chunk']}" for item in similar_chunks)
    print("\n--- RAG context built. Length:", len(rag_context))
    print(f"rag context is {rag_context}")

    # Step 5: Prepare input for agent1-rag agent
    rag_input = f"User question: {query}\n\nContext chunks:\n{rag_context or '(none)'}"

    # Step 6: Run the agent with retry mechanism
    try:
        detailed_ans = await run_agent_with_retry(agent_rag, rag_input)
        print("\n--- Answer from Agent1 ---")
        print(detailed_ans)
    except RuntimeError as err:
        print("Agent failed after retries:", err)
    #agent 2 web search agent 
    try:
        snippets= await web_search(query)
        web_agent_input=f"""User question: {query}

                        Search result snippets:
                        {snippets}

                        """
        web_scrapper_agent_ans=await run_web_agent_with_retry(web_agent,web_agent_input)
        print(f"agent2 response is {web_scrapper_agent_ans}")
    except Exception as e:
        print(f"Agent failed to answer the query: {e}")

    # Run Agent 3: Evaluator
    if detailed_ans and web_scrapper_agent_ans:
        final_ans=await run_evaluator_agent(query, detailed_ans, web_scrapper_agent_ans)
        print(f"FINAL ANS: {final_ans}")

    else:
        print("Skipping evaluation due to missing answers from one or both agents.")


if __name__ == "__main__":
    asyncio.run(main())
