from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key is missing in the environment variables.")

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Hugging Face API token is missing in the environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index

index_name = "medicalbot"
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )



# Load data and split into chunks
try:
    extracted_data = load_pdf_file(data="Data/")  # Ensure this path is correct
    text_chunks = text_split(extracted_data)
except Exception as e:
    raise RuntimeError(f"Error in processing PDF data: {e}")

# Download embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone vector store
try:
    docsearch = LC_Pinecone.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )
    print("Documents successfully embedded and indexed!")
except Exception as e:
    raise RuntimeError(f"Error in embedding or indexing documents: {e}")
