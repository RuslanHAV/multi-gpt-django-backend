from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from account.renderers import UserRenderer
from rest_framework import status
import json


from werkzeug.utils import secure_filename

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub

from langchainapp.htmlTemplates import css, bot_template, user_template
from langchainapp.prompts import set_prompt
import os


from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import SeleniumURLLoader
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
import langchain
import tiktoken

# Create your views here.
TEMP=0.5
MODEL='gpt-3.5-turbo'
PERSONALITY='general assistant'
global conversation

class LibForEmbedding:
        
    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    def get_conversation_chain(vectorstore, temp, model):
        llm = ChatOpenAI(temperature=temp, model_name=model)
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )
        return conversation_chain

    def get_url_text(url):
        responseData = requests.get(url)
        content = responseData.text
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()
        for script in soup(["script", "style"]):
            script.extract()

        cleaned_text_content = soup.get_text()
        return text
    
    # Single PDF
    def get_pdf_text(pdf):
        text = ""
        print('get_pdf_text = ', pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def get_pdfs_text(pdf_docs):
        text = ""
        print('pdf_docs = ', pdf_docs)
        for pdf in pdf_docs:
            text_temp = ""
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text_temp += page.extract_text()
            text += text_temp
        return text


    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks


    def count_tokens(text: str) -> int:
        encoding = tiktoken.get_encoding(OPENAI_EMBEDDING_ENCODING)
        tokens = encoding.encode(text)
        num_tokens = len(tokens)
        return num_tokens, tokens


#!  registerEmbeddingPDF
class EmbeddingURL(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        input_data = request.data
        url = input_data['site_url']
        user_promts = input_data['userPrompts']
        prompt = set_prompt(PERSONALITY)
        request.session["conversation"] = None
        raw_text = LibForEmbedding.get_url_text(url)
        text_chunks = LibForEmbedding.get_text_chunks(raw_text)
        vectorstore = LibForEmbedding.get_vectorstore(text_chunks)
        request.session.conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)

        return Response({"token": '123', "message": "" }, status=status.HTTP_201_CREATED)    
    
    
class EmbeddingPDF(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        input_data = request.data
        print('request.session = ', request.session )
        if request.session.has_key('conversation'):
            print('request.session.conversation = ', request.session.conversation)
        file = request.data.getlist('newFile[]')
        prompt = set_prompt(PERSONALITY)
        request.session["conversation"] = None
        raw_text = LibForEmbedding.get_pdfs_text(file)
        text_chunks = LibForEmbedding.get_text_chunks(raw_text)
        vectorstore = LibForEmbedding.get_vectorstore(text_chunks)
        
        request.session.conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)

        
        return Response({"status": 'success', "data": vectorstore }, status=status.HTTP_201_CREATED)    

class CHAT(APIView):
    
    def post(self, request, format=None):
        input_data = request.data
        user_promts = input_data['userPrompts']
        prompt = set_prompt(PERSONALITY)
        print('request.session = ', request.session.conversation )
        if request.session.has_key('conversation'):
            print('request.session.conversation = ', request.session.conversation)
        # LibForEmbedding.get_conversation_chain(
        #     user_promts,temp=TEMP, model=MODEL)
        conversation_result = request.session.conversation(
        {'question': (prompt+user_promts)})
        return Response({"status": 'success', "data": prompt}, status=status.HTTP_201_CREATED)    
