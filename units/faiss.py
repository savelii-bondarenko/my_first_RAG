from langchain_community.vectorstores import FAISS

def create_FAISS(chunks, embeddings):
    return FAISS.from_texts(
        chunks,
        embeddings
    )

