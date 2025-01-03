from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.embedder.ollama import OllamaEmbedder
from phi.agent import Agent
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.vectordb.search import SearchType
from phi.vectordb.lancedb import LanceDb
from phi.model.groq import Groq
import os
from typing import Optional
from rich.prompt import Prompt
import typer
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model=Groq(api_key=api_key,id="llama-3.1-70b-versatile")
# db_url = "https://arxiv.org/pdf/2311.04934"
db_url = "/tmp/lancedb"
embedder = OllamaEmbedder(model="nomic-embed-text", dimensions=768)

vector_db = LanceDb(
    table_name="recipes",
    embedder=embedder,
    uri="/tmp/lancedb",
    search_type=SearchType.keyword,
)
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

knowledge_base.load(recreate=True)

storage = SqlAgentStorage(table_name="new_recipes", db_file="data.db")
storage.create()

def lancedb_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        run_id=run_id,
        user_id=user,
        storage=storage,
        model=model,
        knowledge=knowledge_base,
        show_tool_calls=True,
        read_chat_history=True,
        add_history_to_messages=True,

        num_history_responses=3,
        debug_mode=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)


if __name__ == "__main__":
    typer.run(lancedb_agent)
# response = agent.("What is the first step of making Gluai Buat Chi from the knowledge base?")
# print("Agent Response:")
# print(response)



# def pdf_assistant(new:bool=False,user:str='user'):
#     run_id:Optional[str]=None
#     if not new:
#         existing_run_ids: List[str]=storage.get_all_run_ids(user)
#         if len(existing_run_ids)>0:
#             run_id=existing_run_ids[0]
#
#     assistant = Assistant(
#         run_id=run_id,  # use any unique identifier to identify the run
#         user_id="user",  # user identifier to identify the user
#
#         knowledge=knowledge_base,
#         storage=storage,
#         read_chat_history=True,
#         show_tool_calls=True,
#         search_knowledge=True,  # Enable debug mode for additional information
#     )
#     if run_id is None:
#         run_id = assistant.run_id
#         print(f"Started run_id: {run_id}\n")
#     else:
#         print(f"Continuing run_id: {run_id}\n")
#     assistant.cli_app(markdown=True)
# if __name__ == "__main__":
#     typer.run(pdf_assistant)