# src/run_search_sota.py
import asyncio
from google.genai import types as gen_types
from google.adk.runners import InMemoryRunner

from iML.tools.adk_search_sota import make_search_sota_root_agent

async def main():
    # Replace this with your pipeline's task_summary when integrating.
    task_summary = """Task: Tabular Regression
Predict median house value from California housing features..."""

    root_agent = make_search_sota_root_agent(task_summary=task_summary, k=3)

    runner = InMemoryRunner(agent=root_agent, app_name="sota-search")

    # Send a simple message to trigger the agent; the instruction uses state.
    user_msg = gen_types.Content(parts=[gen_types.Part(text="run")], role="user")

    async for event in runner.run_async(new_message=user_msg):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None):
                    print(part.text)  # should print a single-element JSON array

if __name__ == "__main__":
    asyncio.run(main())
