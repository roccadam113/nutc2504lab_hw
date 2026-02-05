from HW.tools.search_searxng import search_searxng as ss
from HW.tools.vlm_read_website import vlm_read_website as vlmrweb
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
import json
import os

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0,
)


class State(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
    query: str
    time_range: str
    url: str
    title: str
    json_path: str
    cache_answer: str
    cache_hit: bool
    info_enough: bool


def check_cache(state: State):
    prompt = [
        SystemMessage(
            content=(
                "你是快取索引器。"
                "請把使用者問題濃縮成一個可用來查快取的索引詞。"
                "規則："
                "1 盡量用英文小寫關鍵字，空白分隔"
                "2 6 到 12 個詞以內"
                "3 不要輸出解釋，只輸出 JSON"
                '輸出格式：{"key":"..."}'
            )
        ),
        HumanMessage(content=question),
    ]

    msg = llm.invoke(prompt)
    text = (msg.content or "").strip()

    path = state.get("json_path")
    if not path:
        return {"cache_hit": False}
    if not os.path.exists(path):
        return {"cache_hit": False}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cached_answer = data.get("answer")
        if not cached_answer:
            return {"cache_hit": False}
        return {"cache_hit": True, "cached_answer": cached_answer}
    except Exception as e:
        return {"cache_hit": False}


def planner(state: State):
    pass


def final_answer(state: State):
    pass


def query_gen(state: State):
    pass


def search_tool(state: State):
    pass


def cache_router(state: State) -> Literal["planner", "final_answer"]:
    if state.get("cache_hit"):
        return "final_answer"
    return END


def planner_router(state: State) -> Literal["query_gen", "final_answer"]:
    if state.get("info_enough"):
        return "final_answer"
    return "query_gen"


flow = StateGraph(State)
flow.add_node("check_cache", check_cache)
flow.add_node("planner", planner)
flow.add_node("final_answer", final_answer)
flow.add_node("query_gen", query_gen)
flow.add_node("search_tool", search_tool)
flow.set_entry_point("check_cache")
flow.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "planner": "planner",
        END: END,
    },
)
flow.add_conditional_edges(
    "planner",
    planner_router,
    {
        "query_gen": "query_gen",
        "final_answer": "final_answer",
    },
)
flow.add_edge("query_gen", "search_tool")
flow.add_edge("search_tool", "planner")
flow.add_edge("final_answer", END)

app = flow.compile()
print(app.get_graph().draw_ascii())
