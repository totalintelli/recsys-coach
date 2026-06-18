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
        ("system", (
            "다음 문서를 참고하여 질문에 답하세요.\n"
            "각 문장 끝에 해당 내용의 출처를 공백 한 칸을 두고 [document.pdf 3 page] 같은 형식으로 반드시 붙이세요. "
            "출처를 알 수 없는 문장에는 태그를 붙이지 마세요.\n\n{context}"
        )),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        parts = []
        for doc in docs:
            src = doc.metadata.get("source", "")
            page = doc.metadata.get("page")
            if src and page is not None:
                tag = f"[{src} {page + 1} page]"
            elif src:
                tag = f"[{src}]"
            else:
                tag = ""
            parts.append(f"{tag}\n{doc.page_content}" if tag else doc.page_content)
        return "\n\n".join(parts)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return {"answer": answer, "sources": source_docs}
