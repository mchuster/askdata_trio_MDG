import os
import glob
import time
from pathlib import Path
from pypdf import PdfReader
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY não encontrada no arquivo .env!")

client = genai.Client(api_key=api_key)

# Modelo gratuito de embeddings do Google AI Studio
EMBEDDING_MODEL = "gemini-embedding-001"


def extrair_texto_pdf(caminho_pdf: str) -> list[dict]:
    """Le um arquivo PDF e extrai o texto pagina a pagina com metadados."""
    reader = PdfReader(caminho_pdf)
    paginas = []
    nome_arquivo = Path(caminho_pdf).name

    for idx, pagina in enumerate(reader.pages):
        texto = pagina.extract_text() or ""
        if texto.strip():
            paginas.append({
                "texto": texto.strip(),
                "arquivo": nome_arquivo,
                "pagina": idx + 1
            })
    return paginas


def extrair_texto_markdown(caminho_md: str) -> list[dict]:
    """Le um arquivo Markdown e extrai o texto com metadados."""
    nome_arquivo = Path(caminho_md).name
    with open(caminho_md, "r", encoding="utf-8") as f:
        texto = f.read().strip()

    if texto:
        return [{
            "texto": texto,
            "arquivo": nome_arquivo,
            "pagina": 1
        }]
    return []


def criar_chunks(documentos_paginas: list[dict], chunk_size: int = 700, chunk_overlap: int = 100) -> list[dict]:
    """Divide os textos em blocos com sobreposicao (overlap) para manter contexto."""
    chunks = []

    for item in documentos_paginas:
        texto = item["texto"]
        inicio = 0
        chunk_idx = 1

        while inicio < len(texto):
            fim = inicio + chunk_size
            trecho = texto[inicio:fim]

            chunk_id = f"{item['arquivo']}_p{item['pagina']}_c{chunk_idx}"
            chunks.append({
                "id": chunk_id,
                "texto": trecho,
                "arquivo": item["arquivo"],
                "pagina": item["pagina"],
                "chunk_idx": chunk_idx
            })

            inicio += (chunk_size - chunk_overlap)
            chunk_idx += 1

    return chunks


def indexar_no_chromadb(chunks: list[dict], path_db: str = "./chroma_db", collection_name: str = "askdata_knowledge"):
    """Gera embeddings e salva os chunks e metadados no ChromaDB local persistente."""
    chroma_client = chromadb.PersistentClient(path=path_db)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Total de chunks a serem indexados: {len(chunks)}")

    for i, ch in enumerate(chunks, 1):
        res = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=ch["texto"]
        )
        vetor = res.embeddings[0].values

        collection.upsert(
            ids=[ch["id"]],
            embeddings=[vetor],
            documents=[ch["texto"]],
            metadatas={
                "arquivo": ch["arquivo"],
                "pagina": ch["pagina"],
                "chunk_idx": ch["chunk_idx"]
            }
        )

        time.sleep(0.5)
        if i % 5 == 0 or i == len(chunks):
            print(f"  -> Indexados {i}/{len(chunks)} chunks...")

    print(f"Ingestao concluida com sucesso no ChromaDB ({path_db})! Total salvo: {collection.count()} chunks.")


if __name__ == "__main__":
    pasta_dados = "./data"
    todos_documentos = []

    # 1. Carregar PDFs da pasta data
    for pdf_path in glob.glob(f"{pasta_dados}/*.pdf"):
        print(f"Processando PDF: {pdf_path}")
        todos_documentos.extend(extrair_texto_pdf(pdf_path))

    # 2. Carregar Markdowns da pasta data
    for md_path in glob.glob(f"{pasta_dados}/*.md"):
        print(f"Processando Markdown: {md_path}")
        todos_documentos.extend(extrair_texto_markdown(md_path))

    if not todos_documentos:
        print("Nenhum arquivo PDF ou Markdown encontrado em ./data! Adicione arquivos na pasta para testar.")
    else:
        # 3. Gerar Chunks
        lista_chunks = criar_chunks(todos_documentos, chunk_size=700, chunk_overlap=100)
        # 4. Indexar no ChromaDB
        indexar_no_chromadb(lista_chunks)