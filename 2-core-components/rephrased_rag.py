import bs4
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-4o")

# load the docs
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# split the docs
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# build the vectorstore in-memory and return a retriever
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
retriever = vectorstore.as_retriever()


def rephrase_question(question):
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()
    return rephrase_chain.invoke({"question": question})


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


rephrase_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant tasked with rephrasing questions to improve search results. 
    Please rephrase the following question to make it more specific and searchable:
    
    Original question: {question}
    
    Rephrased question:"""
)

main_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    
    Question: {question}
    Context: {context}
    
    Answer:"""
)


def rag_chain_with_rephrasing(question: str) -> str:
    docs = retriever.invoke(question)

    print(docs)
    # If no relevant documents found, rephrase and try again
    if not docs:
        rephrased_question = rephrase_question(question)
        docs = retriever.invoke(rephrased_question)

    context = format_docs(docs)

    chain = (
        {"context": lambda _: context, "question": lambda _: question}
        | main_prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({})


# Example usage
result = rag_chain_with_rephrasing("What is love?")
print(result)
