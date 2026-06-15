# 🤖 RAG API — Retrieval-Augmented Generation com LangChain + Groq

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal)
![License](https://img.shields.io/badge/license-MIT-green)

Sistema de perguntas e respostas inteligente baseado em documentos PDF, utilizando a técnica **RAG (Retrieval-Augmented Generation)**. O sistema indexa documentos, transforma o conteúdo em vetores semânticos e responde perguntas em linguagem natural com base exclusivamente no conteúdo dos documentos carregados.

---

## ✨ Funcionalidades

- 📄 Carregamento e indexação automática de documentos PDF
- 🔍 Busca semântica por similaridade com ChromaDB
- 🧠 Respostas geradas pelo modelo Llama 3.3 via Groq (gratuito)
- 🌐 API REST com FastAPI e documentação Swagger automática
- 💬 Interface interativa via terminal
- ⚡ Embeddings locais com HuggingFace (sem custo)

---

## 🏗️ Arquitetura

```
Documentos PDF
      ↓
  Chunking (1500 chars, overlap 200)
      ↓
  Embeddings — all-MiniLM-L6-v2 (HuggingFace)
      ↓
  ChromaDB (banco de vetores local)
      ↓
  Busca semântica (top-6 chunks)
      ↓
  Prompt enriquecido com contexto
      ↓
  LLM — Llama 3.3 70B via Groq
      ↓
  Resposta baseada no documento
```

---

## 🛠️ Stack utilizada

| Ferramenta | Função |
|---|---|
| [LangChain](https://langchain.com) | Orquestração do pipeline RAG |
| [Groq + Llama 3.3 70B](https://groq.com) | LLM gratuito e de alta velocidade |
| [ChromaDB](https://trychroma.com) | Banco de vetores local |
| [HuggingFace Embeddings](https://huggingface.co) | Modelo de embeddings local |
| [FastAPI](https://fastapi.tiangolo.com) | API REST com Swagger automático |
| [Python-dotenv](https://pypi.org/project/python-dotenv) | Gerenciamento de variáveis de ambiente |

---

## 📋 Pré-requisitos

- Python 3.10 ou superior
- Conta gratuita no [Groq](https://console.groq.com) para obter a API Key

---

## 🚀 Instalação e uso

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/rag-api.git
cd rag-api
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
GROQ_API_KEY=sua_chave_aqui
```

Obtenha sua chave gratuita em: https://console.groq.com/keys

### 5. Adicione seus documentos PDF

Coloque os arquivos `.pdf` que deseja indexar na pasta `docs/`:

```
docs/
└── seu_documento.pdf
```

### 6. Execute

**Modo terminal (interativo):**
```bash
python main.py
```

**Modo API:**
```bash
uvicorn api:app --reload
```

Acesse a documentação interativa em: http://localhost:8000/docs

---

## 🌐 Endpoints da API

### `GET /`
Verifica se a API está online.

**Resposta:**
```json
{
  "status": "online",
  "mensagem": "RAG API funcionando!"
}
```

### `POST /perguntar`
Envia uma pergunta sobre os documentos carregados.

**Body:**
```json
{
  "pergunta": "Qual é a conclusão da pesquisa?"
}
```

**Resposta:**
```json
{
  "pergunta": "Qual é a conclusão da pesquisa?",
  "resposta": "De acordo com os documentos, a conclusão indica que..."
}
```

---

## 📁 Estrutura do projeto

```
rag-api/
├── docs/                  # PDFs indexados pelo RAG
├── chroma_db/             # Banco de vetores (gerado automaticamente)
├── venv/                  # Ambiente virtual (não versionado)
├── main.py                # Interface interativa via terminal
├── api.py                 # API REST com FastAPI
├── requirements.txt       # Dependências do projeto
├── .env                   # Variáveis de ambiente (não versionado)
├── .gitignore             # Arquivos ignorados pelo Git
└── README.md              # Este arquivo
```

---

## ⚙️ Configurações avançadas

| Parâmetro | Valor atual | Descrição |
|---|---|---|
| `chunk_size` | 1500 | Tamanho de cada pedaço do documento |
| `chunk_overlap` | 200 | Sobreposição entre chunks |
| `k` (retriever) | 6 | Número de chunks recuperados por consulta |
| Modelo LLM | `llama-3.3-70b-versatile` | Modelo utilizado para geração |
| Modelo embedding | `all-MiniLM-L6-v2` | Modelo local de embeddings |

---

## 🔒 Segurança

- A `GROQ_API_KEY` nunca deve ser versionada. Ela é carregada via `.env`, que está no `.gitignore`.
- O banco de vetores `chroma_db/` é gerado localmente e também está no `.gitignore`.

---

## 📄 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

```
MIT License

Copyright (c) 2025 Márcio Belloni

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👨‍💻 Autor

**Márcio Belloni**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Márcio%20Belloni-blue?logo=linkedin)](https://linkedin.com/in/marcio-belloni)
[![GitHub](https://img.shields.io/badge/GitHub-devbelloni-black?logo=github)](https://github.com/devbelloni)

---

## 🗺️ Roadmap

- [ ] Suporte a múltiplos formatos (DOCX, TXT, CSV)
- [ ] Interface web com Streamlit
- [ ] Suporte a múltiplas coleções de documentos
- [ ] Autenticação na API
- [ ] Deploy no Railway ou Render
- [ ] RAG com imagens (multimodal)

---

*Versão 1.0.0 — Junho 2025*