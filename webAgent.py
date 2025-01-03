from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools

load_dotenv()
import os
from dotenv import load_dotenv
load_dotenv()

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
    name="Fiannnacial Agent",
    # model=OpenAIChat(id="gpt-4o"),
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],

    instructions=["use table to display data."],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent = Agent(
    team=[web_agent, fin_agent],
    instructions=["Always include sources","use table to display data"],
    show_tool_calls=True,
    markdown=True
)
multi_ai_agent.print_response("Summarize analyst recommendation of latest news of TSLA", stream=True)
