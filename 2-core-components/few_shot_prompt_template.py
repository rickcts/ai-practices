from langchain_openai import ChatOpenAI
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

#
few_shot = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Question: {input}",
    input_variables=["input"],
)

chain = few_shot | model


def chatbot(topic):
    response = chain.invoke(
        {"input": [{"role": "user", "content": topic}]})
    print(response)
    return response.content


iface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="LangChain Chatbot",
    description="A simple chatbot using LangChain components with Gradio frontend."
)

iface.launch()
