# chroma_utils.py
# Este arquivo contém funções para interagir com o armazenamento vetorial Chroma,
# incluindo indexação e exclusão de documentos.

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
import logging

# Configura o logging para este módulo.
logging.basicConfig(filename='app.log', level=logging.INFO)

# Inicializa o separador de texto e a função de embedding.
# chunk_size: O tamanho máximo de cada pedaço de texto.
# chunk_overlap: A quantidade de sobreposição entre pedaços consecutivos.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
# OpenAIEmbeddings é usado para gerar embeddings vetoriais para os documentos.
embedding_function = OpenAIEmbeddings()

# Inicializa o armazenamento vetorial Chroma.
# persist_directory: Onde os dados do Chroma serão armazenados no disco.
# embedding_function: A função usada para gerar embeddings para os documentos.
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
    # Determina o carregador apropriado com base na extensão do arquivo.
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        # Registra um erro se o tipo de arquivo não for suportado.
        logging.error(f"Tipo de arquivo não suportado: {file_path}")
        raise ValueError(f"Tipo de arquivo não suportado: {file_path}")

    # Carrega o documento usando o carregador selecionado.
    documents = loader.load()
    # Divide os documentos carregados em pedaços usando o text_splitter.
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
        # Carrega e divide o documento.
        splits = load_and_split_document(file_path)

        # Adiciona o metadado 'file_id' a cada pedaço do documento.
        # Isso permite que os pedaços sejam associados ao registro original do documento.
        for split in splits:
            split.metadata['file_id'] = file_id
            logging.info(f"Adicionando metadado file_id {file_id} ao pedaço.")

        # Adiciona os pedaços do documento ao armazenamento vetorial Chroma.
        vectorstore.add_documents(splits)
        logging.info(f"Documento do arquivo {file_path} (ID: {file_id}) indexado com sucesso no Chroma.")
        return True
    except Exception as e:
        # Registra qualquer erro que ocorra durante a indexação.
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
        # Tenta obter os documentos para verificar se existem.
        # Note: A função 'get' do Chroma pode retornar um dicionário com 'ids' e 'documents'.
        docs = vectorstore.get(where={"file_id": file_id})
        logging.info(f"Encontrados {len(docs.get('ids', []))} pedaços de documento para file_id {file_id} no Chroma.")
        print(f"Encontrados {len(docs.get('ids', []))} pedaços de documento para file_id {file_id}")

        # Exclui os documentos do Chroma com base no metadado 'file_id'.
        # Acessa diretamente a coleção subjacente para a operação de exclusão 'where'.
        vectorstore._collection.delete(where={"file_id": file_id})
        logging.info(f"Todos os documentos com file_id {file_id} excluídos do Chroma.")
        print(f"Todos os documentos com file_id {file_id} excluídos do Chroma.")

        return True
    except Exception as e:
        # Registra qualquer erro que ocorra durante a exclusão.
        logging.error(f"Erro ao excluir documento com file_id {file_id} do Chroma: {str(e)}")
        print(f"Erro ao excluir documento com file_id {file_id} do Chroma: {str(e)}")
        return False
