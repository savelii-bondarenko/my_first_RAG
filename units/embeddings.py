from langchain_huggingface import HuggingFaceEmbeddings
import os
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("MODEL_FOR_EMBEDDINGS")
    )
    return embeddings
