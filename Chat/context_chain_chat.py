from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
import os 
from dotenv import load_dotenv
# get current directory
path = os.getcwd()
 
# get parent directory
par_path = os.path.abspath(os.path.join(path, os.pardir))
dotenv_path = par_path+"/.env"
load_dotenv(dotenv_path = dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

chat_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        ( "system", "{context}" ),
        ( "human", "{question}" ),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

current_weather = """
    {
        "surf": [
            {"beach": "Fistral", "conditions": "6ft waves and offshore winds"},
            {"beach": "Polzeath", "conditions": "Flat and calm"},
            {"beach": "Watergate Bay", "conditions": "3ft waves and onshore winds"}
        ]
    }"""

response = chat_chain.invoke(
    {
        "context": current_weather,
        "question": "What is the weather like on Watergate Bay?",
    }
)

print(response)