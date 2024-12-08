from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

if not all([PINECONE_API_KEY, OPENAI_API_KEY, HUGGINGFACEHUB_API_TOKEN]):
    raise EnvironmentError("API keys are not properly set in the .env file.")

# Download embeddings
embeddings = download_hugging_face_embeddings()

# Index configuration
index_name = "medicalbot"

# Load the existing index
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the LLM
llm = OpenAI(temperature=0.4, max_tokens=500)

# Define the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create document and retrieval chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form("msg")
    if not msg:
        return jsonify({"error": "Message is required"}), 400

    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "Sorry, I couldn't process your request.")
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
