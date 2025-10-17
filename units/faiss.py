from langchain_community.vectorstores import FAISS
import os

def create_FAISS(chunks, embeddings):
    return FAISS.from_texts(
        chunks,
        embeddings
    )

