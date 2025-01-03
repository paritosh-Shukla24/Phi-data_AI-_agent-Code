from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import os
import phi.api
import phi
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv
load_dotenv()

phi.api=os.getenv('PHI_API_KEY')
api_key=os.environ['OPENAI_API_KEY']
groq=os.environ['GROQ_API_KEY']

web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    # model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

fin_agent = Agent(
    name="Financial Agent",
    # model=OpenAIChat(id="gpt-4o"),
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],

    instructions=["use table to display data."],
    show_tool_calls=True,
    markdown=True
)

app=Playground(agents=[web_agent, fin_agent]).get_app()
if __name__ == '__main__':
    serve_playground_app("playground:app", reload=True)