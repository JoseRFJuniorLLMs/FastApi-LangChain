# main.py
# Este é o ponto de entrada da nossa aplicação FastAPI.
# Ele define as rotas da API e orquestra a interação entre os diferentes componentes.

from fastapi import FastAPI, File, UploadFile, HTTPException
# Importações relativas para módulos dentro do mesmo pacote 'src'
from .pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from .langchain_utils import get_rag_chain
from .db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, \
    delete_document_record
from .chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
import shutil

# Configura o logging para a aplicação.
# Os logs serão gravados no arquivo 'app.log' com nível INFO.
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializa a aplicação FastAPI.
app = FastAPI()


@app.post("/chat", response_model=QueryResponse)
async def chat(query_input: QueryInput):
    """
    Endpoint para interações de chat com o sistema RAG.

    Args:
        query_input (QueryInput): Objeto contendo a pergunta do usuário,
                                   ID da sessão (opcional) e o modelo a ser usado.

    Returns:
        QueryResponse: Objeto contendo a resposta gerada, ID da sessão e o modelo usado.

    Raises:
        HTTPException: Se ocorrer um erro durante o processamento do chat.
    """
    # Gera um ID de sessão se não for fornecido.
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info(
        f"ID da Sessão: {session_id}, Pergunta do Usuário: {query_input.question}, Modelo: {query_input.model.value}")

    try:
        # Recupera o histórico de chat para a sessão atual.
        chat_history = get_chat_history(session_id)
        # Obtém a cadeia RAG configurada com o modelo especificado.
        rag_chain = get_rag_chain(query_input.model.value)

        # Invoca a cadeia RAG para gerar uma resposta.
        # O 'input' é a pergunta do usuário e 'chat_history' fornece o contexto da conversa.
        answer = rag_chain.invoke({
            "input": query_input.question,
            "chat_history": chat_history
        })['answer']

        # Insere o log da interação no banco de dados.
        insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
        logging.info(f"ID da Sessão: {session_id}, Resposta da IA: {answer}")

        # Retorna a resposta formatada.
        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)
    except Exception as e:
        # Captura e registra quaisquer exceções, retornando um erro HTTP 500.
        logging.error(f"Erro no endpoint /chat para a sessão {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar sua solicitação: {e}")


@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...)):
    """
    Endpoint para upload e indexação de documentos.

    Args:
        file (UploadFile): O arquivo a ser carregado.

    Returns:
        dict: Mensagem de sucesso e o ID do arquivo se a indexação for bem-sucedida.

    Raises:
        HTTPException: Se o tipo de arquivo não for suportado ou se a indexação falhar.
    """
    allowed_extensions = ['.pdf', '.docx', '.html']
    # Obtém a extensão do arquivo e a converte para minúsculas.
    file_extension = os.path.splitext(file.filename)[1].lower()

    # Verifica se a extensão do arquivo é permitida.
    if file_extension not in allowed_extensions:
        logging.warning(f"Tentativa de upload de tipo de arquivo não suportado: {file.filename}")
        raise HTTPException(status_code=400,
                            detail=f"Tipo de arquivo não suportado. Tipos permitidos: {', '.join(allowed_extensions)}")

    # Cria um caminho temporário para salvar o arquivo carregado.
    temp_file_path = f"temp_{uuid.uuid4()}_{file.filename}"

    try:
        # Salva o arquivo carregado em um arquivo temporário.
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Arquivo temporário salvo em: {temp_file_path}")

        # Insere um registro do documento no banco de dados e obtém o ID do arquivo.
        file_id = insert_document_record(file.filename)
        logging.info(f"Registro do documento inserido no DB com ID: {file_id}")

        # Indexa o documento no armazenamento vetorial Chroma.
        success = index_document_to_chroma(temp_file_path, file_id)

        if success:
            logging.info(f"Arquivo {file.filename} (ID: {file_id}) carregado e indexado com sucesso.")
            return {"message": f"Arquivo {file.filename} foi carregado e indexado com sucesso.", "file_id": file_id}
        else:
            # Se a indexação falhar, exclui o registro do documento do banco de dados.
            delete_document_record(file_id)
            logging.error(f"Falha ao indexar {file.filename} (ID: {file_id}). Registro removido do DB.")
            raise HTTPException(status_code=500, detail=f"Falha ao indexar {file.filename}.")
    except Exception as e:
        logging.error(f"Erro durante o upload e indexação do arquivo {file.filename}: {e}")
        # Tenta excluir o registro do documento se ele foi inserido antes da falha.
        if 'file_id' in locals():
            delete_document_record(file_id)
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao processar o upload: {e}")
    finally:
        # Garante que o arquivo temporário seja removido, mesmo que ocorra um erro.
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logging.info(f"Arquivo temporário removido: {temp_file_path}")


@app.get("/list-docs", response_model=list[DocumentInfo])
async def list_documents():
    """
    Endpoint para listar todos os documentos indexados.

    Returns:
        list[DocumentInfo]: Uma lista de objetos DocumentInfo, cada um representando um documento.
    """
    logging.info("Solicitação para listar todos os documentos.")
    try:
        documents = get_all_documents()
        logging.info(f"Retornando {len(documents)} documentos.")
        return documents
    except Exception as e:
        logging.error(f"Erro ao listar documentos: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao listar documentos: {e}")


@app.post("/delete-doc")
async def delete_document(request: DeleteFileRequest):
    """
    Endpoint para excluir um documento do sistema.
    O documento é removido do Chroma e do banco de dados.

    Args:
        request (DeleteFileRequest): Objeto contendo o ID do arquivo a ser excluído.

    Returns:
        dict: Mensagem de sucesso ou erro.

    Raises:
        HTTPException: Se a exclusão falhar no Chroma ou no banco de dados.
    """
    file_id = request.file_id
    logging.info(f"Solicitação para excluir documento com file_id: {file_id}")

    try:
        # Tenta excluir o documento do Chroma.
        chroma_delete_success = delete_doc_from_chroma(file_id)

        if chroma_delete_success:
            logging.info(f"Documento com file_id {file_id} excluído do Chroma.")
            # Se a exclusão do Chroma for bem-sucedida, tenta excluir do banco de dados.
            db_delete_success = delete_document_record(file_id)
            if db_delete_success:
                logging.info(f"Documento com file_id {file_id} excluído do banco de dados.")
                return {"message": f"Documento com file_id {file_id} excluído com sucesso do sistema."}
            else:
                logging.error(
                    f"Excluído do Chroma, mas falha ao excluir documento com file_id {file_id} do banco de dados.")
                # Retorna um erro 500 se a exclusão do DB falhar.
                raise HTTPException(status_code=500,
                                    detail=f"Excluído do Chroma, mas falha ao excluir documento com file_id {file_id} do banco de dados.")
        else:
            logging.error(f"Falha ao excluir documento com file_id {file_id} do Chroma.")
            # Retorna um erro 500 se a exclusão do Chroma falhar.
            raise HTTPException(status_code=500, detail=f"Falha ao excluir documento com file_id {file_id} do Chroma.")
    except Exception as e:
        logging.error(f"Erro geral ao excluir documento com file_id {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro ao excluir o documento: {e}")

# Para executar a aplicação, você usaria: uvicorn src.main:app --reload
# Certifique-se de que o uvicorn esteja instalado.