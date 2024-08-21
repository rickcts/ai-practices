from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import gradio as gr

model = ChatOpenAI(model="gpt-4o")

prompt = PromptTemplate(
    input_variables=["input_text"],
    template="You are a jolly assistant. Here is the user's input: {input_text}. Provide a hilarious concise response."
)


def chatbot(input_text):
    formatted_prompt = prompt.format(input_text=input_text)
    response = model.invoke(formatted_prompt)
    return response.content


iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="LangChain Chatbot",
    description="A simple chatbot using LangChain components with Gradio frontend."
)

iface.launch()
