from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from uuid import uuid4
import os 
from dotenv import load_dotenv

SESSION_ID = str(uuid4())
print(f"Session ID: {SESSION_ID}")

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
embedding_provider = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    graph=graph,
    index_name="moviePlots",
    embedding_node_property="plotEmbedding",
    text_node_property="plot",
)

plot_retriever = RetrievalQA.from_llm(
    llm=llm,
    retriever=movie_plot_vector.as_retriever()
)

response = plot_retriever.invoke(
    {"query": "A movie where a mission to the moon goes wrong"}
)

print(response)