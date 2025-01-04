import os
from openai import OpenAI
from langchain_openai import ChatOpenAI

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(model='gpt-4o',temperature=0.1)
client = OpenAI()

class VideoTranscriber:
    def Trasnsriber(self,file_path):
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcription
        

