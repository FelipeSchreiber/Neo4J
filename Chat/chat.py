from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  
from dotenv import load_dotenv
import os

# get current directory
path = os.getcwd()
 
# get parent directory
par_path = os.path.abspath(os.path.join(path, os.pardir))
dotenv_path = par_path+"/.env"
load_dotenv(dotenv_path = dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

chat_llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY
)

instructions = SystemMessage(content="""
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.
""")

question = HumanMessage(content="What is the weather like?")

response = chat_llm.invoke([
    instructions,
    question
])

print(response.content)