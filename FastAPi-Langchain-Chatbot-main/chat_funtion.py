from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os
# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model='gpt-4o',temperature=0.1)
client = OpenAI()
class Chatbot:
    def create_and_get_chat_response(self,vectorstore, user_query):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        
        template = """You are an AI Q&A chatbot assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Helpful Answer:
        """

        QA_PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        
        # Create the conversational retrieval chain
        chat_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_PROMPT})
        response = chat_chain(user_query)
        chat_history = []
        chat_history.append((user_query, response['result']))
        return response['result'],chat_history
    