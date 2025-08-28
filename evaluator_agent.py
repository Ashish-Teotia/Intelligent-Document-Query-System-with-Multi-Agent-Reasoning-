#evaluator_agent.py
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



# Evaluator agent definition
agent_eval = AssistantAgent(
    name="agent_eval",
    model_client=model_client,
    system_message="""
You are Agent 3 (Evaluator).

You are given:
- A user question
- Agent 1's answer (RAG-based)
- Agent 2's answer (Web-based)

Evaluate:
1. Correctness
2. Completeness
3. Freshness of the information



CHOICE: Agent_1 or Agent_2  
Rationale: (2–3 clear and concise sentences)  
FINAL ANSWER: <restated best answer using your choice>
In the output strictly give FINAL ANSWER ONLY which is best answer of your choice.
"""
)

MAX_EVAL_RETRIES=5

async def run_evaluator_agent(user_query, agent1_output, agent2_output):
    eval_input = f"""User question: {user_query}

Agent 1 (RAG-based) answer:
{agent1_output}

Agent 2 (Web-based) answer:
{agent2_output}

"""
    for attempt in range(1, MAX_EVAL_RETRIES + 1):
        try:
            print(f"\n--- Running Evaluator Agent (Attempt {attempt}) ---")
            task = await agent_eval.run(task=eval_input)

            if task and task.messages and task.messages[-1].content:
                print("\nEvaluator Agent Output:")
                print(task.messages[-1].content)
                return task.messages[-1].content
            else:
                print("No valid content returned. Retrying...")

        except Exception as e:
            print(f"Exception on attempt {attempt}: {e}")

        await asyncio.sleep(1)

    raise RuntimeError("Evaluator agent failed after maximum retries.")
    


