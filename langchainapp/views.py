from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from account.renderers import UserRenderer
from rest_framework import status
import json
from django.http import HttpResponse

from langchainapp.LangChainAttrForm import LangChainAttrForm
from werkzeug.utils import secure_filename

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from django_thread import Thread

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA

from langchain.chains.query_constructor.base import AttributeInfo

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
import pinecone
import slack

from slackeventsapi import SlackEventAdapter

from .models import LangChainAttr

import atexit
import queue
import threading

from django.core.mail import mail_admins


def _worker():
    while True:
        func, args, kwargs = _queue.get()
        try:
            res = func(*args, **kwargs)
        except:
            print('error')
            import traceback
            details = traceback.format_exc()
            print(details)
            mail_admins('Background process exception', details)
        finally:
            _queue.task_done()  # so we can join at exit

def postpone(func):
    def decorator(*args, **kwargs):
        _queue.put((func, args, kwargs))
    return decorator

_queue = queue.Queue()
_thread = threading.Thread(target=_worker)
_thread.daemon = True
_thread.start()

def _cleanup():
    _queue.join()   # so we don't exit too soon

atexit.register(_cleanup)

# Create your views here.
TEMP=0.5
MODEL='gpt-3.5-turbo'
PERSONALITY='general assistant'
EMBEDDING_TYPE = ''
EMBEDDING_VAL = ''
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
global conversation

class LibForEmbedding:
        
    def get_vectorstore(text_chunks):
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
            environment=os.environ["PINECONE_ENVIRONMENT"],  # next to api key in console
        )
        
        index_name = "langchaindb"
        
        embeddings = OpenAIEmbeddings()
        metadata = {id: EMBEDDING_VAL, type: EMBEDDING_TYPE}
        vectorstore = Pinecone.from_texts(texts=text_chunks, metadata=metadata, embedding=embeddings, index_name=index_name)
        # vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
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
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def get_pdfs_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            text_temp = ""
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text_temp += page.extract_text()
            text += text_temp
        return text

    def get_txts_text(txt_files):
        text = ""
        for txt in txt_files:
            for line in txt:
                text = text +  str(line.decode())
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
        # user_promts = input_data['userPrompts']
        prompt = set_prompt(PERSONALITY)
        EMBEDDING_VAL = url
        EMBEDDING_TYPE = 'url'
        request.session["conversation"] = None
        raw_text = LibForEmbedding.get_url_text(url)
        text_chunks = LibForEmbedding.get_text_chunks(raw_text)
        vectorstore = LibForEmbedding.get_vectorstore(text_chunks)
        request.session.conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)

        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)    
    
    
class EmbeddingPDF(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        input_data = request.data
        files = request.FILES.getlist('files')

        prompt = set_prompt(PERSONALITY)
        request.session["conversation"] = None
        EMBEDDING_VAL = files
        EMBEDDING_TYPE = 'pdf'
        raw_text = LibForEmbedding.get_pdfs_text(files)
        text_chunks = LibForEmbedding.get_text_chunks(raw_text)
        vectorstore = LibForEmbedding.get_vectorstore(text_chunks)
        
        request.session.conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)

        
        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)


class EmbeddingCSV(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        input_data = request.data
        files = request.FILES.getlist('files')

        prompt = set_prompt(PERSONALITY)
        request.session["conversation"] = None
        raw_text = LibForEmbedding.get_pdfs_text(files)
        text_chunks = LibForEmbedding.get_text_chunks(raw_text)
        vectorstore = LibForEmbedding.get_vectorstore(text_chunks)
        
        request.session.conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)

        
        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)    


class EmbeddingTXT(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        input_data = request.data
        files = request.FILES.getlist('files')
        EMBEDDING_VAL = files   
        EMBEDDING_TYPE = 'txt'
        prompt = set_prompt(PERSONALITY)
        request.session["conversation"] = None
        raw_text = LibForEmbedding.get_txts_text(files)
        text_chunks = LibForEmbedding.get_text_chunks(raw_text)
        vectorstore = LibForEmbedding.get_vectorstore(text_chunks)
        
        request.session.conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)

        
        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)    
    

class CHAT(APIView):
    
    def post(self, request, format=None):
        input_data = request.data
        user_promts = input_data['userPrompts']
        prompt = set_prompt(PERSONALITY)
        history = []
        if 'history' in input_data:
            history = input_data['history']
        MODEL = input_data['modal']
        
        
        stripped_user_promps = user_promts.strip()
        index = pinecone.Index(PINECONE_INDEX_NAME)
        embedding = OpenAIEmbeddings()
        vectorstore = Pinecone(index, embedding.embed_query, "text")
        conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)
        temp = [tuple((row[0], row[1]) for row in history)]
        
        conversation_result = conversation(
        {'question': (prompt+user_promts), "chat_history": temp})
        history.append(tuple(([user_promts, conversation_result['answer']])))
        
        return Response({"status": 'success', "answer": conversation_result['answer'], "history" :history }, status=status.HTTP_201_CREATED)    

class LangAttr(APIView):
    
    def post(self, request, format=None):
        input_data = request.data
        open_ai_key = input_data['openAIKey']
        prompts = input_data['prompts']
        save_data = None
        if save_data is None:
            save_data = {}
        save_data['attribute'] = ''
        save_data['attr_type'] = ''
        save_data['is_active'] = 1
        
        if open_ai_key != "":
            save_data['attribute'] = open_ai_key
            save_data['attr_type'] = 'open_ai_key'
            save_data['is_active'] = 1
            
        if prompts != "":
            save_data['attribute'] = prompts
            save_data['attr_type'] = 'prompts'
            save_data['is_active'] = 1
            
        if open_ai_key == "" and prompts == ""  :
            return Response({"message": "Invalid input parameter" }, status=status.HTTP_201_CREATED)    
        form = LangChainAttrForm(save_data)  
        se = LangChainAttrForm(save_data)  
        if form.is_valid(): 
            try: 
                form.save()  
                return Response({"status": "success", "message": "save successfully"}, status=status.HTTP_201_CREATED)  
            except:  
                pass  
        form = LangChainAttrForm()  
        return Response({"status": "success", "message": "save successfully"}, status=status.HTTP_201_CREATED)      

class LangSlack(APIView):

    @postpone
    def conversationWithLang(self, text):
        SLACK_TOKEN=os.environ["SLACK_TOKEN"]
        SIGNING_SECRET=os.environ["SIGNING_SECRET"]
        prompt = set_prompt(PERSONALITY)
        history = []
        MODEL = 'gpt-3.5-turbo'
        stripped_user_promps = text.strip()
        index = pinecone.Index(PINECONE_INDEX_NAME)
        embedding = OpenAIEmbeddings()
        vectorstore = Pinecone(index, embedding.embed_query, "text")
        conversation = LibForEmbedding.get_conversation_chain(
            vectorstore, temp=TEMP, model=MODEL)
        conversation_result = conversation(
        {'question': (prompt + text), "chat_history": history})
        client = slack.WebClient(token=SLACK_TOKEN)
        client.chat_postMessage(channel='#multigpt-slackbot',text=conversation_result['answer'])
        return 'success'
        
    
    def post(self, request, format=None):
        input_data = request.data
        
        if 'challenge' in input_data:
            response_data = {}
            challenge = input_data['challenge']
            response_data['challenge'] = challenge
            return HttpResponse(json.dumps(response_data), content_type="application/json")
        else : 
            
            event_callback_type = input_data['event']['type']
            user_message = 'client_msg_id'

            text = input_data['event']['text']
            self.conversationWithLang(text)
        return Response(status=status.HTTP_200_OK)


class GetLangAttr(APIView):
    def post(self, request, format=None):
        input_data = request.data
        

class MailDetect(APIView):
    def post(self, request, format=None):
        input_data = request.data