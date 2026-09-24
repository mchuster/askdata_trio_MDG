# trio_IA_navi

Integrantes: Mateus Huster, Deric Gabriel, German Fros

Dominio do tema escolhido: Cibersegurança e Governança de TI

# Requisitos

- **Python 3.11 ou superior** (o projeto foi desenvolvido e testado com **Python 3.14**)
- pip
- Chave de API do modelo de linguagem utilizado (configurada no `.env`)
- Conexão com a internet na primeira execução (download do modelo de embedding)

## Instalação

### 1. Criar e ativar o ambiente virtual

```bash
python3 -m venv .venv
```

Ative o ambiente:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

A primeira instalação demora, pois o `sentence-transformers` baixa o PyTorch (alguns GB).

| Pacote | Uso |
|---|---|
| `chromadb` | banco vetorial |
| `sentence-transformers` | embeddings multilíngues (PT/ES) |
| `pypdf` | leitura dos PDFs |
| `streamlit` | interface web |
| `python-dotenv` | leitura do `.env` |
| SDK do modelo de linguagem | geração das respostas (adicionar ao `requirements.txt`) |

### 3. Configurar o ambiente

1. Coloque os PDFs na pasta `data/`.
2. Crie o arquivo `.env` na raiz com a chave da API:

```
NOME_DA_VARIAVEL_DA_API=sua-chave-aqui
```

3. Garanta que o `.gitignore` contenha:

```
.venv/
chroma_db/
.env
__pycache__/
```

## Uso

### 1. Indexar os documentos
rodar o comando:
```bash
python src/ingest.py
```

### 2. Testar no terminal

```bash
python src/rag_engine.py
```

### 3. Abrir a interface

```bash
streamlit run src/app.py
```
