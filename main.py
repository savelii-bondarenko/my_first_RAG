import os

import torch
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from units.PDF_reader import read_pdf
from units.chunk_breaker import split_text
from units.embeddings import create_embeddings
from units.faiss import create_FAISS

load_dotenv()

text = read_pdf("HarryPotter.pdf")
chunks = split_text(text)
embeddings = create_embeddings()

vectorstore = create_FAISS(chunks, embeddings)
retriever = vectorstore.as_retriever()

llm = HuggingFacePipeline.from_model_id(
    model_id=os.getenv("MODEL_ID"),
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95
    },
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True
    },
    device_map="auto"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

query = "Что такое философский камень?"
result = qa_chain.invoke({"query": query})

# Выводим результат
print(result['result'])
