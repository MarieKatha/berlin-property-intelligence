import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from pprint import pprint
from tools import get_now, generate_password


load_dotenv() # Load environment variables from .env file

# client = genai.Client()

# response = client.models.generate_content(
#     model="gemini-2.5-flash-lite",
#     contents="What is the capital of France?",
#     # config=types.GenerateContentConfig(
#     #     max_output_tokens=200,
#     #     temperature=0.6
#     # )
# )

model = init_chat_model("gemini-2.5-flash-lite",
                        model_provider="google_genai",
                        max_tokens=200,
                        temperature=1
                        )

model_with_tools = model.bind_tools(
    [
        get_now,
        generate_password
    ]

)

system_msg = SystemMessage(
    """
    You are a helpful assistant. You adapt the language to the last language
    the user used. You are nice and helpful and proactive and natural
    in conversation. Please always check your avaiable tools before generating
    a response.
    Available tools: get_now, generate_password
    """)
human_msg = HumanMessage("Generier mir ein random secure passwort der länge 12 ohne sonderzeichen?")
messages = [system_msg, human_msg]

response = model_with_tools.invoke(messages)

tools_map = {
    "get_now": get_now,
    "generate_password": generate_password,
}

if response.tool_calls:
    messages.append(response)
    for tool_call in response.tool_calls:
        tool = tools_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        messages.append({"role": "tool", "content": str(result), "tool_call_id": tool_call["id"]})

    final_response = model_with_tools.invoke(messages)
    print(final_response.__dict__)
else:
    print(response.__dict__)

if __name__ == '__main__':
    pass
