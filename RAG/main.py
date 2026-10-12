import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer."""
)


def format_docs(docs):
    """Combine retrieved documents into one string."""
    return "\n\n".join(doc.page_content for doc in docs)


# 1: RAG without LCEL
def retrieval_chain_without_lcel(query: str):
    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = prompt_template.format_messages(
        context=context,
        question=query,
    )

    response = llm.invoke(messages)
    return response.content


# 2: RAG with LCEL
def create_retrieval_chain_with_lcel():
    return (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    query = "What is Pinecone in machine learning?"

    # normal RAG 
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nRAG without LCEL:")
    print(result_without_lcel)

    # RAG with LCEL
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke(
        {"question": query}
    )
    print("\nRAG with LCEL:")
    print(result_with_lcel)