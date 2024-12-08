from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# Extract data from the PDF file
def load_pdf_file(data):
    try:
        loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents
    except Exception as e:
        raise RuntimeError(f"Error loading PDFs from directory {data}: {e}")

# Split the data into text chunks
def text_split(extracted_data):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(extracted_data)
        return text_chunks
    except Exception as e:
        raise RuntimeError(f"Error splitting text into chunks: {e}")

# Download embeddings from Hugging Face
def download_hugging_face_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Error downloading Hugging Face embeddings: {e}")
