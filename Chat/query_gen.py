from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
import os 
from dotenv import load_dotenv

# get current directory
path = os.getcwd()
 
# get parent directory
par_path = os.path.abspath(os.path.join(path, os.pardir))
dotenv_path = par_path+"/.env"
load_dotenv(dotenv_path = dotenv_path)

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""

cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"],
)

cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True,
    allow_dangerous_requests=True
)

cypher_chain.invoke({"query": "What is the plot of the movie Toy Story?"})