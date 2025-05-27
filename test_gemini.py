# test_gemini.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

try:
    print("Tentando inicializar ChatGoogleGenerativeAI...")
    # ATUALIZE ESTA LINHA PARA USAR O MODELO CORRETO
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    print("ChatGoogleGenerativeAI inicializado com sucesso!")

    print("\nTentando inicializar GoogleGenerativeAIEmbeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("GoogleGenerativeAIEmbeddings inicializado com sucesso!")

    # Se você quiser, pode tentar uma chamada simples para verificar a autenticação
    print("\nTentando invocar o LLM com uma pergunta de teste...")
    response = llm.invoke("Qual é a capital da França?")
    print(f"Resposta do LLM: {response.content}")

    print("\nTodos os componentes do Google Generative AI foram carregados e testados com sucesso.")

except Exception as e:
    print(f"\nERRO: Falha ao carregar ou usar componentes do Google Generative AI.")
    print(f"Detalhes do erro: {e}")
    print("\nVerifique se sua variável de ambiente GOOGLE_APPLICATION_CREDENTIALS (ou GOOGLE_API_KEY) está definida corretamente e se 'langchain-google-genai' está instalado.")
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and not os.getenv("GOOGLE_API_KEY"):
        print("Nenhuma das variáveis de ambiente GOOGLE_APPLICATION_CREDENTIALS ou GOOGLE_API_KEY foi encontrada.")
