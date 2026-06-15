from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(
    title="RAG API",
    description="API de perguntas e respostas baseada em documentos PDF",
    version="1.0.0"
)

# Carrega tudo uma vez quando a API sobe
def inicializar_rag():
    documentos = []
    for arquivo in os.listdir("docs"):
        if arquivo.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join("docs", arquivo))
            documentos.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado em analisar documentos técnicos e acadêmicos.
Responda a pergunta baseado APENAS no contexto abaixo.
Se realmente não houver nenhuma informação relevante, diga que não encontrou.

Contexto:
{context}

Pergunta: {question}

Resposta detalhada:
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = inicializar_rag()

# Modelo da requisição
class Pergunta(BaseModel):
    pergunta: str

# Modelo da resposta
class Resposta(BaseModel):
    pergunta: str
    resposta: str

@app.get("/")
def root():
    return {"status": "online", "mensagem": "RAG API funcionando!"}

@app.post("/perguntar", response_model=Resposta)
def perguntar(body: Pergunta):
    if not body.pergunta.strip():
        raise HTTPException(status_code=400, detail="Pergunta não pode ser vazia")
    
    resposta = rag_chain.invoke(body.pergunta)
    return Resposta(pergunta=body.pergunta, resposta=resposta)