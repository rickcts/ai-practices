from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
import gradio as gr


class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field


parser = PydanticOutputParser(pydantic_object=Joke)

model = ChatOpenAI(model="gpt-4", temperature=0.7,
                   max_tokens=256, frequency_penalty=0.0)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = prompt | model.with_structured_output(Joke)


def chatbot(topic):
    response: Joke = chain.invoke(
        {"query": [{"role": "user", "content": f"Generate a joke about {topic}"}]}
    )

    return response.setup, response.punchline


def format_joke(setup, punchline):
    return f"{setup}\n\n{punchline}"


with gr.Blocks() as iface:
    gr.Markdown("# LangChain Joke Generator")
    gr.Markdown(
        "A simple joke generator using LangChain components with Gradio frontend.")

    topic_input = gr.Textbox(label="Enter a topic for the joke")
    joke_output = gr.Accordion(label="Joke")
    with joke_output:
        setup_output = gr.Textbox(label="Setup")
        punchline_output = gr.Textbox(label="Punchline")

    generate_button = gr.Button("Generate Joke")
    generate_button.click(
        fn=chatbot,
        inputs=topic_input,
        outputs=[setup_output, punchline_output]
    )

iface.launch()
