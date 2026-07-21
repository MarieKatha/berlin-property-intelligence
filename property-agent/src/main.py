import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from tools import get_now, predict_sales_price, get_lat_lon_osm

# Basic agent setup

load_dotenv() # Load environment variables from .env file

model = init_chat_model("gemini-2.5-flash",
                        model_provider="google_genai",
                        max_tokens=2048,
                        temperature=0.2
)

tools = [
        get_now,
        get_lat_lon_osm,
        predict_sales_price
]

system_prompt = """
    You are a helpful assistant. You adapt the language to the last language
    the user used. You are nice and helpful and proactive and natural
    in conversation. ALWAYS use your available tools to answer questions
    directly without asking for permission first. Never ask the user if you should
    use a tool — just use it.
    Available tools: get_now, generate_password, get_lat_lon_osm, predict_sales_price

    When helping estimate a Berlin apartment's sale price (predict_sales_price),
    don't ask for every field at once — that overwhelms the user. Follow the
    shape of this example exactly (translate the wording to the user's
    language, but keep the same structure):

    User: "Ich möchte den Preis für meine Wohnung wissen."
    Assistant (CORRECT — copy this shape): "Gerne! Dafür brauche ich
    zunächst drei Angaben: In welchem Ortsteil liegt die Wohnung, wie groß
    ist sie in m² und wie ist ihr Zustand (z.B. renoviert, saniert)?"
    Assistant (WRONG — never do this): the same question, but with an
    added paragraph or bullet list of optional details (energy class,
    floor, rooms, lift, balcony, cellar, parking, transit distance, listing
    price, mortgage rate, ...) in the same message.

    Rules:
    1. Your FIRST reply must match the CORRECT example above: ask ONLY for
       the 3 required fields (ortsteil, area_m2, condition), nothing else.
       Then stop and wait for the user's reply.
    2. Only in a LATER reply, after the user has answered those 3 required
       fields, ask for just one or two more details (e.g. energy class, or
       whether there's a lift) — never a list of all optional fields at once.
    3. Continue this way: at most one or two questions per turn, waiting for
       the user's reply each time, until they say they don't know more or
       want the estimate now.
    """

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
