from fastapi import FastAPI
from data_store_pipeline import VideoTranscriber
from upload_doucment import DocumentUploader
from chat_funtion import Chatbot 
from vectorstore_loader import VectorStoreLoader

from langchain_community.document_loaders import PyPDFLoader, TextLoader ,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_chroma import Chroma

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from openai import OpenAI
import os


# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model='gpt-4o',temperature=0.1)
client = OpenAI()




# Create an instance of VectorStoreLoader
loader = VectorStoreLoader()
folder_path = "db"  # Replace with the desired path
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
