#agent1.py
import asyncio
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
import azure_config_list # your config with model_details dict

model_details = azure_config_list.model_details

model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=model_details['deployment_name'],
    model=model_details['model'],
    api_version=model_details['api_version'],
    azure_endpoint=model_details['base_url'],
    api_key=model_details['api_key'],
)

agent_rag = AssistantAgent(
    name="agent_rag",
    model_client=model_client,
    system_message="""
You are Agent 1: a RAG Agent.
Given the user question and material from document chunks, first evaluate chunks.
Then answer using only that context.
Use structured reasoning.Make sure your output should be strictly the final response which you have generated.
""",
)

MAX_RETRIES = 5

async def run_agent_with_retry(agent, rag_input):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Attempt {attempt} to run agent...")
            task1 = await agent.run(task=rag_input)

            if task1 and task1.messages and task1.messages[-1].content:
                detailed_ans = task1.messages[-1].content
                print("Got valid response from agent.")
                return detailed_ans
            else:
                print(f"Attempt {attempt} failed: No valid content in response.")

        except Exception as e:
            print(f"Attempt {attempt} raised exception: {e}")

        await asyncio.sleep(1)

    raise RuntimeError(f"Failed to get valid response after {MAX_RETRIES} attempts.")
