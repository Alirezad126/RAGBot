from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

script_dir = os.path.dirname(__file__)
db_dir = os.path.join(script_dir, 'vectorDB/')
          
def create_embedding_vectordb(docs, embeddings):
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=db_dir)
    return db

def load_embedding_vectordb():
    db = Chroma(embedding_function=embeddings,persist_directory=db_dir)
    return db