import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        provider="novita",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512
    )

CUSTOM_PROMPT_TEMPLATE = """
You are a compassionate and knowledgeable medical help assistant. 
Provide accurate, clear, and supportive information based on the provided context. 
If the question requires urgent medical attention, clearly advise the user to seek immediate help from a healthcare professional or emergency services. 
Do not provide a formal diagnosis or prescribe medications; instead, guide the user with safe, evidence-based advice and next steps. 

Context: {context}

Question: {question}

Start the answer directly. No small talk please
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = (
    {
        "context": db.as_retriever(search_kwargs={"k": 3}),
        "question": RunnablePassthrough()
    }
    | set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    | load_llm(HUGGINGFACE_REPO_ID)
    | StrOutputParser()
)

result = qa_chain.invoke("What is Heart Disease?")

print("Answer:", result)
