import os
from typing import List, Dict, Any
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY não encontrada no arquivo .env!")

# Modelos oficiais gratuitos do Google AI Studio
EMBEDDING_MODEL = "gemini-embedding-001"
MODELO_FLASH = "gemini-3.5-flash-lite"  # Modelo Gemini

class RAGEngine:
    def __init__(self, path_db: str = "./chroma_db", collection_name: str = "askdata_knowledge"):
        """Inicializa a conexão com o ChromaDB e o cliente Gemini."""
        self.client = genai.Client(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path=path_db)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _gerar_embedding(self, texto: str) -> List[float]:
        """Gera o embedding da pergunta do usuário."""
        res = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texto
        )
        return res.embeddings[0].values

    def recuperar_contexto(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Busca no ChromaDB os top_k chunks mais relevantes para a query."""
        vetor_query = self._gerar_embedding(query)
        
        resultados = self.collection.query(
            query_embeddings=[vetor_query],
            n_results=top_k
        )
        
        chunks_recuperados = []
        if resultados and resultados["documents"] and resultados["documents"][0]:
            for doc, meta, dist in zip(
                resultados["documents"][0],
                resultados["metadatas"][0],
                resultados["distances"][0]
            ):
                chunks_recuperados.append({
                    "texto": doc,
                    "arquivo": meta.get("arquivo", "desconhecido"),
                    "pagina": meta.get("pagina", 1),
                    "distancia": dist,
                    "similaridade": round(1.0 - dist, 4)
                })
        return chunks_recuperados

    def responder_pergunta(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Executa o pipeline RAG completo: Retrieval -> Prompt Augmentation -> Generation."""
        # 1. Recuperar contexto do banco vetorial
        chunks = self.recuperar_contexto(query, top_k=top_k)
        
        if not chunks:
            return {
                "resposta": "Nenhum documento encontrado na base de conhecimento. Por favor, processe arquivos primeiro.",
                "fontes": []
            }

        # 2. Formatar o contexto recuperado com marcação de origem
        contexto_formatado = ""
        for idx, ch in enumerate(chunks, 1):
            contexto_formatado += f"\n--- [FONTE {idx} | Arquivo: {ch['arquivo']} | Página: {ch['pagina']}] ---\n"
            contexto_formatado += ch["texto"] + "\n"

        # 3. System Instruction blindado contra alucinações
        system_instruction = """
Você é o 'AskData', um assistente corporativo de inteligência artificial da DataLakers.
Sua missão é responder à pergunta do usuário de forma clara, profissional e EXCLUSIVAMENTE baseada nos trechos de documentos fornecidos no contexto.

REGRAS OBRIGATÓRIAS:
1. Responda apenas com informações presentes no <contexto_recuperado>.
2. Se a resposta NÃO estiver no contexto fornecido, NÃO tente inventar ou usar conhecimentos externos. Responda exatamente: "Desculpe, não encontrei informações sobre isso nos documentos fornecidos."
3. Ao responder, cite o nome do arquivo e a página de onde a informação foi extraída.
4. Mantenha um tom profissional, direto e em bom português.
"""

        # 4. Prompt com delimitadores
        prompt_final = f"""
<contexto_recuperado>
{contexto_formatado}
</contexto_recuperado>

<pergunta_do_usuario>
{query}
</pergunta_do_usuario>
"""

        # 5. Chamada ao Modelo Gemini com temperatura baixa (0.1)
        response = self.client.models.generate_content(
            model=MODELO_FLASH,
            contents=prompt_final,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )

        return {
            "resposta": response.text.strip(),
            "fontes": chunks
        }

if __name__ == "__main__":
    engine = RAGEngine()
    
    print("=" * 60)
    print("TESTE DO MOTOR RAG (Terminal)")
    print("=" * 60)
    
    while True:
        pergunta = input("\nFaça uma pergunta sobre seus documentos ('sair' para encerrar): ").strip()
        if pergunta.lower() in ["sair", "exit"]:
            break
        if not pergunta:
            continue
            
        resultado = engine.responder_pergunta(pergunta, top_k=3)
        
        print("\nRESPOSTA DO ASSISTENTE:")
        print(resultado["resposta"])
        
        print("\nFONTES UTILIZADAS (Metadados do ChromaDB):")
        for f in resultado["fontes"]:
            print(f"  - {f['arquivo']} (Página {f['pagina']}) - Similaridade: {f['similaridade']:.2%}")

# Dica de Engenharia: Se algo nao funcionar de primeira, leia o traceback e debugar faz parte do projeto!