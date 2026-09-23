import chromadb
from collections import defaultdict

# 1. Conectar ao ChromaDB persistente
client = chromadb.PersistentClient(path="./src/chroma_db")
collection = client.get_or_create_collection(
    name="askdata_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# 2. Recuperar todos os documentos e metadados indexados
dados = collection.get(include=["documents", "metadatas"])
ids = dados["ids"]
docs = dados["documents"]
metadatas = dados["metadatas"]

total_chunks = len(ids)

print("=" * 60)
print("RELATORIO DE AUDITORIA: CACA AO CHUNK PERDIDO")
print("=" * 60)
print(f"Total de chunks indexados: {total_chunks}")

if total_chunks == 0:
    print("Colecao vazia! Execute 'python src/ingestion.py' primeiro.")
    exit()

# 3. Calcular tamanho medio dos chunks
tamanhos = [len(doc) for doc in docs]
tamanho_medio = sum(tamanhos) / total_chunks
print(f"Tamanho medio dos chunks: {tamanho_medio:.1f} caracteres")
print(f"Menor chunk: {min(tamanhos)} caracteres | Maior chunk: {max(tamanhos)} caracteres")

# 4. Contar chunks por arquivo e por pagina
chunks_por_arquivo = defaultdict(int)
chunks_por_pagina = defaultdict(lambda: defaultdict(int))

for meta in metadatas:
    arquivo = meta.get("arquivo", "desconhecido")
    pagina = meta.get("pagina", 0)
    chunks_por_arquivo[arquivo] += 1
    chunks_por_pagina[arquivo][pagina] += 1

print("\n--- Distribuicao por Arquivo ---")
for arquivo, qtd in chunks_por_arquivo.items():
    print(f"* {arquivo}: {qtd} chunks")
    for pag, count in sorted(chunks_por_pagina[arquivo].items()):
        print(f"    - Pagina {pag}: {count} chunks")

# 5. Identificar chunks cortados no meio da frase (sem pontuacao final)
pontuacao_final = ('.', '!', '?', ':', '"', "'", '”', '’')
chunks_cortados = []

for cid, doc, meta in zip(ids, docs, metadatas):
    texto_limpo = doc.strip()
    if texto_limpo and not texto_limpo.endswith(pontuacao_final):
        chunks_cortados.append((cid, texto_limpo, meta))

print("\n--- Analise de Cortes de Frase ---")
print(f"Chunks sem pontuacao final: {len(chunks_cortados)} de {total_chunks} ({len(chunks_cortados)/total_chunks*100:.1f}%)")

if chunks_cortados:
    print("\nExemplos de chunks cortados no meio da frase:")
    for cid, texto, meta in chunks_cortados[:3]:
        fim_trecho = texto[-60:].replace("\n", " ")
        print(f"  [ID: {cid}] Arquivo: {meta.get('arquivo')}, Pag: {meta.get('pagina')}")
        print(f"    Final do texto: \"...{fim_trecho}\"")
        print("    " + "-" * 50)

print("\nAuditoria concluida. Discutam no trio se o overlap de 100 caracteres")
print("preserva o contexto semantico ou se os parametros precisam ser calibrados.")
print("=" * 60)