import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() == "true"
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "")


def _get_embeddings():
    from langchain_upstage import UpstageEmbeddings
    return UpstageEmbeddings(
        api_key=UPSTAGE_API_KEY,
        model="solar-embedding-1-large",
    )


def build_vectorstore(docs: list[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = _get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def answer_question(vectorstore: FAISS, question: str) -> dict[str, Any]:
    """{'answer': str, 'sources': list[Document]} 반환"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    source_docs = retriever.invoke(question)

    if OFFLINE_MODE:
        answer = "\n\n---\n\n".join(doc.page_content for doc in source_docs)
        return {"answer": answer, "sources": source_docs}

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_upstage import ChatUpstage

    llm = ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서를 참고하여 질문에 답하세요.\n\n{context}"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return {"answer": answer, "sources": source_docs}
