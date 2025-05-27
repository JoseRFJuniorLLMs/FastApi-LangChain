# pydantic_models.py
# Este arquivo define os modelos Pydantic para validação de dados de requisição e resposta.

from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

# Enum para definir os nomes dos modelos de linguagem permitidos.
class ModelName(str, Enum):
    # Definimos APENAS o modelo que você quer usar.
    GEMINI_2_0_FLASH = "gemini-2.0-flash"

# Modelo para a entrada de uma consulta de chat.
class QueryInput(BaseModel):
    question: str  # A pergunta do usuário (obrigatória).
    session_id: str = Field(default=None)  # ID da sessão (opcional, será gerado se não for fornecido).
    model: ModelName = Field(default=ModelName.GEMINI_2_0_FLASH)  # Modelo de linguagem a ser usado, com padrão.

# Modelo para a resposta de uma consulta de chat.
class QueryResponse(BaseModel):
    answer: str  # A resposta gerada pelo modelo.
    session_id: str  # O ID da sessão.
    model: ModelName  # O modelo usado para gerar a resposta.

# Modelo para informações sobre um documento indexado.
class DocumentInfo(BaseModel):
    id: int  # Identificador único do documento.
    filename: str  # Nome do arquivo do documento.
    upload_timestamp: datetime  # Carimbo de data/hora do upload.

# Modelo para a requisição de exclusão de arquivo.
class DeleteFileRequest(BaseModel):
    file_id: int # O ID do arquivo a ser excluído.