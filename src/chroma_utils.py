# chroma_utils.py
# Este arquivo contém funções para interagir com o armazenamento vetorial Chroma,
# incluindo indexação e exclusão de documentos.

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
import logging

# ADICIONADO: Importar para carregar credenciais de conta de serviço
from google.oauth2 import service_account

# Configura o logging para este módulo.
logging.basicConfig(filename='app.log', level=logging.INFO)

# ADICIONADO: Carregar credenciais da conta de serviço explicitamente
# O caminho para o arquivo credentials.json (assumindo que está na raiz do projeto)
CREDENTIALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.json")

try:
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    logging.info(f"Credenciais carregadas com sucesso de: {CREDENTIALS_FILE}")
except Exception as e:
    logging.error(f"Erro ao carregar credenciais de {CREDENTIALS_FILE}: {e}")
    # Se as credenciais não puderem ser carregadas, a aplicação não deve prosseguir.
    # Você pode querer levantar uma exceção ou lidar com isso de outra forma.
    raise RuntimeError(f"Falha ao carregar credenciais da conta de serviço: {e}")


# Inicializa o separador de texto e a função de embedding.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

# GoogleGenerativeAIEmbeddings é usado para gerar embeddings vetoriais para os documentos.
# Passando as credenciais explicitamente.
embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", credentials=credentials)

# Inicializa o armazenamento vetorial Chroma.
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


def load_and_split_document(file_path: str) -> List[Document]:
    """
    Carrega um documento de um determinado caminho de arquivo e o divide em pedaços.
    Suporta arquivos PDF, DOCX e HTML.

    Args:
        file_path (str): O caminho para o arquivo do documento.

    Returns:
        List[Document]: Uma lista de objetos Document, onde cada um é um pedaço do documento.

    Raises:
        ValueError: Se o tipo de arquivo não for suportado.
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        logging.error(f"Tipo de arquivo não suportado: {file_path}")
        raise ValueError(f"Tipo de arquivo não suportado: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)


def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """
    Carrega, divide e indexa um documento no armazenamento vetorial Chroma.
    Adiciona o file_id como metadado a cada pedaço para rastreamento.

    Args:
        file_path (str): O caminho para o arquivo do documento.
        file_id (int): O ID único associado a este arquivo no banco de dados.

    Returns:
        bool: True se a indexação for bem-sucedida, False caso contrário.
    """
    try:
        splits = load_and_split_document(file_path)

        for split in splits:
            split.metadata['file_id'] = file_id
            logging.info(f"Adicionando metadado file_id {file_id} ao pedaço.")

        vectorstore.add_documents(splits)
        logging.info(f"Documento do arquivo {file_path} (ID: {file_id}) indexado com sucesso no Chroma.")
        return True
    except Exception as e:
        logging.error(f"Erro ao indexar documento {file_path} (ID: {file_id}) no Chroma: {e}")
        print(f"Erro ao indexar documento: {e}")
        return False


def delete_doc_from_chroma(file_id: int) -> bool:
    """
    Exclui todos os pedaços de documento associados a um determinado file_id do Chroma.

    Args:
        file_id (int): O ID do arquivo cujos pedaços devem ser excluídos.

    Returns:
        bool: True se a exclusão for bem-sucedida, False caso contrário.
    """
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        logging.info(f"Encontrados {len(docs.get('ids', []))} pedaços de documento para file_id {file_id} no Chroma.")
        print(f"Encontrados {len(docs.get('ids', []))} pedaços de documento para file_id {file_id}")

        vectorstore._collection.delete(where={"file_id": file_id})
        logging.info(f"Todos os documentos com file_id {file_id} excluídos do Chroma.")
        print(f"Todos os documentos com file_id {file_id} excluídos do Chroma.")

        return True
    except Exception as e:
        logging.error(f"Erro ao excluir documento com file_id {file_id} do Chroma: {str(e)}")
        print(f"Erro ao excluir documento com file_id {file_id} do Chroma: {str(e)}")
        return False
