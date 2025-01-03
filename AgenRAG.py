from phi.agent import Agent
from phi.knowledge.langchain import LangChainKnowledgeBase

from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


import os
from dotenv import load_dotenv
load_dotenv()
API_KEY=os.environ['EXA_API_KEY']
api_key=os.environ['OPENAI_API_KEY']
chroma_db_dir = "./chroma_db"
def load_vector_store():
    state_of_the_union = ws_settings.ws_root.joinpath("data/demo/state_of_the_union.txt")
    # -*- Load the document
    raw_documents = TextLoader(str(state_of_the_union)).load()
    # -*- Split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    # -*- Embed each chunk and load it into the vector store
    FAISS.from_documents(documents, OpenAIEmbeddings(), persist_directory=str(chroma_db_dir))


# -*- Get the vectordb
db = FAISS(embedding_function=OpenAIEmbeddings(), persist_directory=str(chroma_db_dir))
# -*- Create a retriever from the vector store
retriever = db.as_retriever()

# -*- Create a knowledge base from the vector store
knowledge_base = LangChainKnowledgeBase(retriever=retriever)

agent = Agent(knowledge_base=knowledge_base, add_references_to_prompt=True)
conv.print_response("What did the president say about technology?")
