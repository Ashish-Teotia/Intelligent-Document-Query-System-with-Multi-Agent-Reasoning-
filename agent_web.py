# agent_web.py

import requests
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import azure_config_list
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
model_details = azure_config_list.model_details


model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=model_details['deployment_name'],
    model=model_details['model'],
    api_version=model_details['api_version'],
    azure_endpoint=model_details['base_url'],
    api_key=model_details['api_key'],
)
# Retry wrapper for web_search
MAX_RETRIES = 5

async def web_search(query: str) -> str:
    """Search the web using Search1API with retry mechanism."""
    API_URL = "https://api.search1api.com/search"
    data = {
        "query": query,
        "search_service": "google",
        "max_results": 5,
        "crawl_results": 2,
        "image": False,
        "language": "en",
        "time_range": "month"
    }
    headers = {
        "Content-Type": "application/json"
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Web search attempt {attempt} for query: {query}")
            response = requests.post(API_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            results = response.json()

            snippets = []
            if results and results.get("results"):
                top_results = results["results"][:5]
                for idx, r in enumerate(top_results, start=1):
                    snippet = r.get("snippet")
                    if snippet:
                        snippets.append(f"({idx}) {snippet}")
            if snippets:
                print(f"Wev scrapping done propoerly with output {snippets}")
                return "\n".join(snippets)
            else:
                print("No snippets found, retrying...")

        except Exception as e:
            print(f"Attempt {attempt} failed with error: {e}")
            await asyncio.sleep(1)

    raise RuntimeError("Failed to retrieve search results after multiple attempts.")

# Assistant agent using web search tool
web_agent = AssistantAgent(
    name="agent_web",
    model_client=model_client,
    system_message="""
You are Agent 2: a Web Search Agent.

You will receive:
- A user question
- A list of up to 5 web search result snippets (labeled as (1), (2), ...)

Your task:
1. Read all snippets and identify relevant ones.
2. Use only the snippets to answer the user's question.
3. Clearly cite sources (e.g., "According to snippet (2)...").
4. Do NOT use external knowledge.
5. Provide a clear, structured final answer.
"""
)


async def run_web_agent_with_retry(agent, agent_input):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Agent web scrapper attempt {attempt}...")
            task = await agent.run(task=agent_input)

            if task and task.messages and task.messages[-1].content:
                print("Agent returned a valid response.")
                return task.messages[-1].content
            else:
                print("No content in agent response.")
        except Exception as e:
            print(f" Error: {e}")
        await asyncio.sleep(1)

    raise RuntimeError("Agent failed after maximum retries.")