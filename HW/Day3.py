from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from HW_asr import asr
from os import path

WAV_PATH = "HW\Podcast_EP14.wav"
OUT_DIR = "./out"


llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0,
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    wav_path: str
    transcript: str
    minutes: str
    summary: str
    final: str


def Asr(state: AgentState):
    data = asr(state["wav_path"])
    return {"transcript": data["transcript"]}


def Minutes_taker(state: AgentState):
    text = state["transcript"]
    return {"minutes": text}


def Summarizer(content: str):
    """
    會議記錄專業嚴謹摘要工具
    會將文字檔全部内容做一個專業摘要。
    """
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "你是一個資深專業的摘要撰寫者。請從逐字稿整理：重點、決策、代辦事項。用繁體中文輸出。",
            ),
            ("user", "{text}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    response = []
    for c in chain.stream({"text": content}):
        print(c.replace("**", ""), end="", flush=True)
        response.append(c)
    print("end")
    return {"summary": "".join(response)}


def writer(state: Annotated):
    return {"final": f"{state['minutes']}\n\n{state['summary']}"}


def should_continue(state: AgentState):
    if state.get("minutes") and state.get("summary"):
        return "writer"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("asr", Asr)
workflow.add_node("minutes_taker", Minutes_taker)
workflow.add_node("summarizer", Summarizer)
workflow.add_node("writer", writer)

workflow.set_entry_point("asr")

workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")

workflow.add_conditional_edges("minutes_taker", should_continue, {"writer": "writer"})
workflow.add_conditional_edges("summarizer", should_continue, {"writer": "writer"})

workflow.add_edge("writer", END)

app = workflow.compile()
print(app.get_graph().draw_ascii())

result = app.invoke({"wav_path": WAV_PATH})
print(result["final"])
