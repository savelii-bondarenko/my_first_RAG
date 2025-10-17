import os

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_mistralai import ChatMistralAI


def create_RAG_chain(retriever, system_prompt):
    llm = ChatMistralAI(
        model=os.getenv("MODEL_ID"),
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=os.getenv("MODEL_TEMPERATURE"),
        max_tokens=os.getenv("MODEL_LENGTH_TOKENS"),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": system_prompt,
            "document_variable_name": "context",
        },
    )
    return qa_chain

