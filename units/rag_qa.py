from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

def create_RAG_chain(vectorstore):
    tokenizer = AutoTokenizer.from_pretrained(os.getenv("MODEL_ID"))
    model = AutoModelForCausalLM.from_pretrained(os.getenv("MODEL_ID"))

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=os.getenv("MODEL_LENGTH_TOKENS"),
        temperature=os.getenv("MODEL_TEMPERATURE"),
    )

    llm = HuggingFacePipeline(pipe)
    retriever = vectorstore.as_retriever(search_kwargs={"k":4})

    qa_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        chain_type="stuff",
        llm=llm,
    )
    return qa_chain

