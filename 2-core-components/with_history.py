from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
import gradio as gr

model = ChatOpenAI(model="gpt-4o",
                   temperature=0.7,
                   max_tokens=256,
                   frequency_penalty=0.0)

examples = [{"input": "What is the capital of France?",
             "output": "The capital of France is Paris."},
            {"input": "What is the largest city in Japan?",
            "output": "The largest city in Japan is Tokyo."}]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Question: {input}\nAnswer: {output}\n",
)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


#
few_shot = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Question: {input}",
    input_variables=["input"],
)


chain = few_shot | model

with_history = RunnableWithMessageHistory(
    runnable=few_shot,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="output",
)


def chatbot(topic):
    response = with_history.invoke(
        {"input": [{"role": "user", "content": topic}]},
        config={"configurable": {
            "session_id": "test",
        }}
    )
    print(store)  # To check the stored history
    return response.content


iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="LangChain Chatbot",
    description="A simple chatbot using LangChain components with Gradio frontend."
)

iface.launch()
