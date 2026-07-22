import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from pprint import pprint
from tools import (
    get_now,
    predict_sales_price,
    predict_construction_price,
    predict_rentals_price,
    get_lat_lon_osm,
    scrape_property_listing,
)

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
        predict_sales_price,
        predict_construction_price,
        predict_rentals_price,
        scrape_property_listing
]

system_prompt = """
    You are a helpful assistant. You communicate only in English, regardless
    of what language the user writes in -- always reply in English. You are
    nice and helpful and proactive and natural in conversation. ALWAYS use
    your available tools to answer questions directly without asking for
    permission first. Never ask the user if you should use a tool — just
    use it.
    Available tools: get_now, get_lat_lon_osm, scrape_property_listing,
    predict_sales_price, predict_construction_price, predict_rentals_price

    If the user gives you an ImmobilienScout24 (immobilienscout24.de) listing
    URL, call scrape_property_listing on it first instead of asking them to
    type out every field by hand. Its fields come back as raw scraped text
    (e.g. condition "Renoviert", energy_class "A+", has_lift "Ja"/"Nein") --
    translate these into the exact values the matching predict_* tool below
    expects, the same way you'd translate the user's own words. If a
    required field wasn't found on the page, ask the user for just that
    field instead of the whole set. A scraped listing that has a
    `condition` field is a resale/secondary-market apartment -- use
    predict_sales_price for it, never predict_construction_price (new-build
    listings don't have a condition at all).

    predict_sales_price (resale/secondary-market apartments),
    predict_construction_price (new-construction apartments -- no condition
    field) and predict_rentals_price (monthly warm rent) all predict a
    price and share the rule below. Pick the one tool that matches what the
    user is actually asking about (buying/selling an existing apartment vs.
    a new-build vs. renting) — never call more than one for the same
    question.

    Some of these tools' fields only accept German category values, because
    that's the language the underlying model was trained on (e.g. `condition`
    values like renovierungsbedürftig/renoviert/modernisiert/saniert/
    kernsaniert, or `position` values like gartenhaus/hinterhaus/
    seitenflügel/vorderhaus). Never show these German values to the user or
    ask them to pick from this list. Always phrase the question in English
    (e.g. "what's its condition — needs renovation, renovated, modernized,
    refurbished, or fully refurbished?"), then translate the user's English
    answer to the matching German value yourself before calling the tool.

    When helping estimate a price with any of these three tools, don't ask
    for every field at once — that overwhelms the user. Follow the shape of
    this example exactly:

    User: "I'd like to know the price of my apartment."
    Assistant (CORRECT — copy this shape): "Sure! For that I first need
    three things: which neighbourhood (Ortsteil) is it in, how large is it
    in m², and what's its condition (e.g. renovated, refurbished)?"
    Assistant (WRONG — never do this): the same question, but with an
    added paragraph or bullet list of optional details (energy class,
    floor, rooms, lift, balcony, cellar, parking, transit distance, listing
    price, mortgage rate, ...) in the same message.

    Rules:
    1. Your FIRST reply must match the CORRECT example above: ask ONLY for
       the required fields of the matching tool, nothing else — ortsteil,
       area_m2 and condition for predict_sales_price/predict_rentals_price,
       or just ortsteil and area_m2 for predict_construction_price (it has
       no condition field). Then stop and wait for the user's reply.
    2. Only in a LATER reply, after the user has answered those required
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
        query = input("You: ")
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
