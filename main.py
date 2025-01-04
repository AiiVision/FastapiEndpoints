from fastapi import FastAPI
from data_store_pipeline import VideoTranscriber
from upload_doucment import DocumentUploader
from chat_funtion import Chatbot 
from vectorstore_loader import VectorStoreLoader

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from openai import OpenAI
import os


# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model='gpt-4o',temperature=0.1)
client = OpenAI()




# Create an instance of VectorStoreLoader
loader = VectorStoreLoader()
folder_path = "Database"  # Replace with the desired path
# Initialize the vector store
loader.initialize_vectorstore(folder_path)
# Get the initialized vector store
vectorstore = loader.get_vectorstore()
print(vectorstore)

app = FastAPI()

@app.post("/transcribe/")
async def transcribe_video(video_path: str):
    # file_path = "video.mp4"
    transcriber = VideoTranscriber()
    transcription = transcriber.Trasnsriber(video_path)
    return {"transcription": transcription}

@app.post("/upload/")
async def upload_docs(file_paths: list):
    uploader = DocumentUploader()
    # file_paths = ["requirements.docx"]
    uploader.upload_documents(file_paths)
    vectorstore = uploader.get_vectorstore()
    return {"message": "Documents uploaded and vector store updated."}

@app.post("/chat/")
async def chat_response(user_query: str, chat_history: list):
    chatbot = Chatbot()
    # user_query = "What is the summary of the document?" 
    response,chat_history = chatbot.create_and_get_chat_response(vectorstore,user_query)
    return {"response": response, "chat_history": chat_history}


