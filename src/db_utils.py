# db_utils.py
# Este arquivo contém funções para interagir com o banco de dados SQLite,
# gerenciando logs de chat e metadados de documentos.

import sqlite3
from datetime import datetime

DB_NAME = "rag_app.db" # Nome do arquivo do banco de dados SQLite.

def get_db_connection():
    """
    Cria e retorna uma conexão com o banco de dados SQLite.
    Define row_factory para facilitar o acesso aos dados por nome da coluna.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    """
    Cria a tabela 'application_logs' se ela ainda não existir.
    Esta tabela armazena o histórico de chat e as respostas do modelo.
    """
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def create_document_store():
    """
    Cria a tabela 'document_store' se ela ainda não existir.
    Esta tabela mantém um registro dos documentos carregados.
    """
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    """
    Insere um novo registro de log de chat na tabela 'application_logs'.
    """
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    """
    Recupera o histórico de chat para uma determinada session_id.
    Retorna uma lista de mensagens formatadas para uso pelo sistema RAG.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

def insert_document_record(filename):
    """
    Insere um novo registro de documento na tabela 'document_store'.
    Retorna o ID do arquivo recém-inserido.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    """
    Exclui um registro de documento da tabela 'document_store' com base no file_id.
    Retorna True se a exclusão for bem-sucedida.
    """
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents():
    """
    Recupera todos os registros de documentos da tabela 'document_store'.
    Retorna uma lista de dicionários, cada um representando um documento.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    # Converte os objetos Row em dicionários para facilitar a serialização/uso.
    return [dict(doc) for doc in documents]

# Inicializa as tabelas do banco de dados quando o módulo é carregado.
create_application_logs()
create_document_store()
