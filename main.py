import os
from typing import TypedDict, Annotated   #in lg, we need state- shared memory, all nodes will pick data from same state or shared memory
import operator  #any new msg will come after the list of old msg, they will get appended to the old list
#this helps in context building for ai

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver  #pgs-we need a long term memory, so to connect lag application to our postgresql, this package we use
from langchain_core.messages import (   #importing fro this package
    AnyMessage,  #stores all kind of msg
    HumanMessage,  #msg done by user is represented by this
    AIMessage,     #ai agent's reply to user handled by this class
    SystemMessage,  #prompt given to system
)

from langchain_groq import ChatGroq   #since we use model from groq 

from tools.tavily_tool import tavily_search  #these made by me inside tools folder
from tools.flight_tool import search_flights
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)
DATABASE_URL=os.getenv("DATABASE_URL")
#creating state , in every langgraph app we make a state , all agents take data from this state
class TravelState(TypedDict):    #class created , in form of  dict, we store data here
    messages: Annotated[list[AnyMessage], operator.add]  #any msg from user will come in a list
    user_query: str  #format of storing data , our query will get stored into user_query key and this key becomes i/p to another agent
    flight_results: str #o/p of flight agent
    hotel_results: str #o/p of hotel agent is stored in this key and its i/p is user_query
    itinerary: str #it goes to state nd takes o/p of hotel and flight agents nd then creates its o/p 
    llm_calls: int #int format of data input , how many times model is called to check for cost 
     # Flight Agent
def flight_agent(state: TravelState):   #travelstate-shared memory
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content=f"Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
    # Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
def itinerary_agent(state: TravelState):

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    response = llm.invoke([   #llm is used here
        SystemMessage(
            content="You are an expert travel planner"   #telling the llm what it has to be 
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,  #itinerary is updated
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
# Final Response Agent
def final_agent(state: TravelState):

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
graph = StateGraph(TravelState)  #importing graph and providing travel state to it

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

#establishing connection
graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)  #start and end nodes inbuilt in langgraph


# Persistent connection so both CLI and Streamlit can share the compiled app
_conn = psycopg.connect(DATABASE_URL, autocommit=True)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_shreyashi"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)

