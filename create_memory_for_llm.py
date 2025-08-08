from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
    
documents = load_pdf_files(DATA_PATH)
# print(f"Loaded {len(documents)} documents from {DATA_PATH}")

# Step 2: Split documents into smaller chunks
def create_text_splitter(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust chunk size as needed
        chunk_overlap=50,  # Adjust overlap as needed
        # length_function=len,
    )
    
    text_chunks = text_splitter.split_documents(extracted_data)
    
    return text_chunks

text_chunks = create_text_splitter(extracted_data=documents)
# print(f"Created {len(text_chunks)} text chunks from the documents.")


# Step 3: Create embeddings for the text chunks
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

get_embedding = get_embedding_model()

# Step 4: Create a vector store from the text chunks and embeddings FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db= FAISS.from_documents(text_chunks,get_embedding) 
db.save_local(DB_FAISS_PATH)