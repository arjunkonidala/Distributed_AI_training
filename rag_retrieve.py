# rag/retriever.py

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_db = Chroma(
    collection_name="clinical_guidelines",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


def retrieve_guidelines(query: str, k: int = 4):
    docs = vector_db.similarity_search(query, k=k)

    return [
        doc.page_content
        for doc in docs
    ]
