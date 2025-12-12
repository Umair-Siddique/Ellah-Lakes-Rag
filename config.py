import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("INDEX_NAME")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

