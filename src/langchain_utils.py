# langchain_utils.py
# Este arquivo implementa o núcleo do sistema RAG usando LangChain,
# configurando o modelo de linguagem, retriever e a cadeia RAG.

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import List
from langchain_core.documents import Document
import os
# ADICIONADO: Importar o módulo logging
import logging
# LINHA CORRIGIDA: Adicionado o ponto '.' para importação relativa
from .chroma_utils import vectorstore  # Importa a instância do vectorstore do chroma_utils.

# ADICIONADO: Importar para carregar credenciais de conta de serviço
from google.oauth2 import service_account

# ADICIONADO: Carregar credenciais da conta de serviço explicitamente
# O caminho para o arquivo credentials.json (assumindo que está na raiz do projeto)
CREDENTIALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "credentials.json")

try:
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE)
    logging.info(f"Credenciais carregadas com sucesso de: {CREDENTIALS_FILE}")
except Exception as e:
    logging.error(f"Erro ao carregar credenciais de {CREDENTIALS_FILE}: {e}")
    raise RuntimeError(f"Falha ao carregar credenciais da conta de serviço: {e}")

# O retriever é configurado para buscar os 2 documentos mais relevantes (k=2).
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# O StrOutputParser é usado para parsear a saída do modelo de linguagem para uma string.
output_parser = StrOutputParser()

# Prompt do sistema para contextualizar a pergunta do usuário.
contextualize_q_system_prompt = (
    "Dado um histórico de chat e a última pergunta do usuário "
    "que pode fazer referência a um contexto no histórico do chat, "
    "formule uma pergunta independente que possa ser compreendida "
    "sem o histórico do chat. NÃO responda à pergunta, "
    "apenas reformule-a se necessário e, caso contrário, retorne-a como está."
)

# Template de prompt para contextualizar a pergunta.
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Template de prompt para a cadeia de perguntas e respostas.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente de IA útil. Use o seguinte contexto para responder à pergunta do usuário."),
    ("system", "Contexto: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


def get_rag_chain(model: str = "ggemini-2.0-flash"):
    """
    Cria e retorna a cadeia RAG (Retrieval-Augmented Generation).

    Args:
        model (str): O nome do modelo de linguagem a ser usado (padrão: "gemini-2.0-flash").

    Returns:
        Runnable: A cadeia RAG completa pronta para ser invocada.
    """
    # Inicializa o modelo de linguagem de chat do Google Generative AI (Gemini).
    # Passando as credenciais explicitamente.
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.7, credentials=credentials)

    # Cria um retriever ciente do histórico.
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Cria uma cadeia de documentos que combina os documentos recuperados com a pergunta
    # para gerar uma resposta usando o LLM e o qa_prompt.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Cria a cadeia RAG completa.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
