from typing import Annotated, Literal, TypedDict
import gradio as gr
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
import requests
from bs4 import BeautifulSoup


@tool
# 建立一個搜尋工具工具給 ChatGPT
def search(query: str) -> str:
    """Call to surf the web."""
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://duckduckgo.com/html/?q={query}"
    response = requests.get(search_url, headers=headers)

    soup = BeautifulSoup(response.text, 'html.parser')

    results = soup.find_all('a', class_='result__a', limit=5)

    search_results = []
    for result in results:
        title = result.get_text()
        url = result['href']
        snippet = result.find_next('a', class_='result__snippet').get_text(
        ) if result.find_next('a', class_='result__snippet') else "No snippet available."
        search_results.append({
            'title': title,
            'url': url,
            'snippet': snippet
        })

    result_str = "\n\n".join(
        [f"Title: {item['title']}\nURL: {item['url']}\nSnippet: {item['snippet']}" for item in search_results])
    return result_str


# 將工具放入工具節點
tools = [search]
tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7,
                   max_tokens=256).bind_tools(tools)

# 確認 Agent 是否繼續


def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}


# 建立工作流
workflow = StateGraph(MessagesState)

# 添加工作流節點
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 設定工作流入口
workflow.set_entry_point("agent")

# 設定 Agent 會決定使用 should_continue 來確認是否繼續
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 設定 tools 完成後會回傳給 Agent
workflow.add_edge("tools", 'agent')


def process_query(query: str) -> str:
    # Create a new instance of the workflow for each query
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    # Invoke the workflow with the user's query
    final_state = app.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": 42}}
    )

    # Return the agent's response
    return final_state["messages"][-1].content


# Create the Gradio interface
iface = gr.Interface(
    fn=process_query,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="AI Agent with Web Search Capability",
    description="Ask a question, and the AI agent will use web search if needed to provide an answer."
)

iface.launch()
