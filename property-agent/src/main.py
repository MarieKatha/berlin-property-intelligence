import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from tools import get_now, generate_password

# Basic agent setup

load_dotenv() # Load environment variables from .env file

model = init_chat_model("gemini-2.5-flash",
                        model_provider="google_genai",
                        max_tokens=2048,
                        temperature=1
)

tools = [
        get_now,
        generate_password
]

# model_with_tools = model.bind_tools(tools)

system_prompt = """
    You are a helpful assistant. You adapt the language to the last language
    the user used. You are nice and helpful and proactive and natural
    in conversation. ALWAYS use your available tools to answer questions
    directly without asking for permission first. Never ask the user if you should
    use a tool — just use it.
    Available tools: get_now, generate_password
    """
# messages = [system_prompt, human_msg]

# response = model_with_tools.invoke(messages)

memory = MemorySaver()

agent_executor = create_agent(
    model,
    tools,
    checkpointer=memory,
    system_prompt=system_prompt
)

# query = "Welcher Tag ist heute?"
# query = "Generier mir ein random secure passwort der länge 12 ohne sonderzeichen?"
query = "Wie heiße ich?"

config = {"configurable": {"thread_id": "1"}}

# if response.tool_calls:
#     messages.append(response)
#     for tool_call in response.tool_calls:
#         tool = tools_map[tool_call["name"]]
#         result = tool.invoke(tool_call["args"])
#         messages.append({"role": "tool", "content": str(result), "tool_call_id": tool_call["id"]})

#     final_response = model_with_tools.invoke(messages)
#     print(final_response.__dict__)
# else:
#     print(response.__dict__)

def extract_content(msg):
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return " ".join(
            block.get("text", "")
            for block in msg.content
            if isinstance(block, dict)
        )
    return str(msg.content)

if __name__ == '__main__':
    while True:
        query = input("Du: ")
        if query.lower() in ("exit", "quit"):
            break
        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode="values",
        ):
            msg = step["messages"][-1]
            if isinstance(msg, AIMessage):
                text = extract_content(msg)
                if text:
                    print(f"AI: {text}")
            else:
                msg.pretty_print()
