from dotenv import load_dotenv
from units.PDF_reader import read_pdf
from units.chunk_breaker import split_text
from units.embeddings import create_embeddings
from units.faiss import create_FAISS
from units.rag_chain import create_RAG_chain
from units.create_system_prompt import adjust_system_prompt
load_dotenv()

text = read_pdf("HarryPotter.pdf")
chunks = split_text(text)
embeddings = create_embeddings()

vectorstore = create_FAISS(chunks, embeddings)
retriever = vectorstore.as_retriever()

system_prompt = (
    "Ти — розумний асистент, який відповідає українською мовою. "
    "Будь ввічливим і пояснюй просто."
    "Не роби виділення тексту"
)

system_prompt = adjust_system_prompt(system_prompt)

qa_chain = create_RAG_chain(retriever, system_prompt)

query = "Що таке філосовський камінь?"
result = qa_chain.invoke({"query": query})

print(result['result'])
