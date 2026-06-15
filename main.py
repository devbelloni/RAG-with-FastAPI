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

def carregar_documentos(caminho_pasta: str):
    documentos = []
    for arquivo in os.listdir(caminho_pasta):
        if arquivo.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(caminho_pasta, arquivo))
            documentos.extend(loader.load())
    print(f"{len(documentos)} páginas carregadas.")
    return documentos

def criar_vectorstore(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,   # era 500 — chunks maiores = mais contexto por pedaço
        chunk_overlap=200  # era 50 — mais sobreposição = não perde informação nas bordas
    )
    chunks = splitter.split_documents(documentos)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Apaga o vectorstore antigo para reindexar com os novos chunks
    import shutil
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    print(f"{len(chunks)} chunks indexados.")
    return vectorstore

def criar_rag(vectorstore):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}  # era 3 — busca mais chunks para ter mais contexto
    )

    prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado em analisar documentos técnicos e acadêmicos.
Responda a pergunta baseado APENAS no contexto abaixo.
Se a informação estiver no contexto, mesmo que parcialmente, use-a para responder.
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

if __name__ == "__main__":
    documentos = carregar_documentos("docs")
    vectorstore = criar_vectorstore(documentos)
    rag = criar_rag(vectorstore)

    print("\nRAG pronto! Digite sua pergunta (ou 'sair' para encerrar):")
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() == "sair":
            break
        resposta = rag.invoke(pergunta)
        print(f"\nRAG: {resposta}")