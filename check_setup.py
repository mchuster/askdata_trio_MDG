import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def test_imports():
    print("[1/3] Testando imports de bibliotecas essenciais...")
    modules = ["google.genai", "chromadb", "pypdf", "dotenv"]
    all_ok = True
    for mod in modules:
        try:
            __import__(mod)
            print(f"  [OK] Modulo '{mod}' importado com sucesso.")
        except ImportError as e:
            print(f"  [FALHA] Nao foi possivel importar '{mod}': {e}")
            all_ok = False
    return all_ok


def test_api_key():
    print("\n[2/3] Verificando chave de API...")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  [FALHA] Variavel GEMINI_API_KEY nao encontrada no .env ou ambiente.")
        return False

    mascarada = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"  [OK] GEMINI_API_KEY configurada ({mascarada}).")
    return True


def test_data_folder():
    print("\n[3/3] Validando documentos na pasta data/...")
    data_dir = Path("data")
    if not data_dir.exists() or not data_dir.is_dir():
        print("  [FALHA] Diretorio 'data/' nao encontrado.")
        return False

    files = [f for f in data_dir.iterdir() if not f.name.startswith(".")]
    if not files:
        print("  [AVISO] Nenhum arquivo encontrado dentro de 'data/'.")
        return False

    import pypdf
    pdf_count = 0
    total_pages = 0

    for file_path in sorted(files):
        if file_path.suffix.lower() == ".pdf":
            pdf_count += 1
            try:
                reader = pypdf.PdfReader(str(file_path))
                pages = len(reader.pages)
                total_pages += pages
                print(f"  [OK] PDF: {file_path.name} - {pages} paginas.")
            except Exception as e:
                print(f"  [FALHA] Erro ao ler {file_path.name}: {e}")
        elif file_path.suffix.lower() in [".md", ".txt"]:
            print(f"  [OK] Documento de texto: {file_path.name}")
        else:
            print(f"  [INFO] Outro arquivo: {file_path.name}")

    print(f"\nResumo: {pdf_count} PDFs validados, totalizando {total_pages} paginas.")
    if pdf_count < 3 or pdf_count > 5:
        print(f"  [AVISO] O trio deve ter entre 3 e 5 arquivos PDF (atualmente: {pdf_count}).")
    return True


if __name__ == "__main__":
    print("=== SMOKE TEST DO TRIO (AskData) ===")
    ok_imports = test_imports()
    ok_key = test_api_key()
    ok_data = test_data_folder()

    if ok_imports and ok_key and ok_data:
        print("\n[SUCESSO] Ambiente validado com sucesso em todos os requisitos!")
        sys.exit(0)
    else:
        print("\n[ERRO] Ajuste as pendencias acima antes de avancar.")
        sys.exit(1)