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
import langchain

from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders.csv_loader import CSVLoader

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
from django.db.models import Q
from django.db.models import F
from django.core.mail import mail_admins
from datetime import datetime, timedelta

import imaplib
import smtplib
from email.parser import Parser
from email.message import EmailMessage
from email.header import decode_header

open_ai_key = LangChainAttr.objects.filter(Q(attr_type='open_ai_key')).latest('created_at')

def sanitize_header(header):
    decoded_header = decode_header(header)
    sanitized_header = []
    for value, encoding in decoded_header:
        if isinstance(value, bytes):
            value = value.decode(encoding)
        sanitized_value = value.replace('\r', '').replace('\n', '')
        sanitized_header.append(sanitized_value)
    return ''.join(sanitized_header)

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



class SaveData:
    @staticmethod
    def save_attr(id):
        save_data = None
        if save_data is None:
            save_data = {}
        save_data['attribute'] = id
        save_data['attr_type'] = 'document'
        save_data['is_active'] = 1
        
        form = LangChainAttrForm(save_data)  
        se = LangChainAttrForm(save_data)  
        if form.is_valid(): 
            try: 
                form.save()  
                return 'ok'  
            except:  
                pass  
        form = LangChainAttrForm() 

class LibForEmbedding:
    def get_vectorstore(text_chunks, file_id):
        embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key.attribute)

        # PineCone
        # pinecone.init(
        #     api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
        #     environment=os.environ["PINECONE_ENVIRONMENT"],  # next to api key in console
        # )
        # index_name = PINECONE_INDEX_NAME
        # metadata = {id: embedding_val, type: 'document'}
        # vectorstore = Pinecone.from_texts(texts=[str(chunk) for chunk in text_chunks], metadata=metadata, embedding=embeddings, index_name=index_name)

        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local(
            folder_path = f"./store/", 
            index_name = file_id
        )

        return vectorstore

    def get_conversation_chain(vectorstore, temp, model):
        llm = ChatOpenAI(openai_api_key=open_ai_key.attribute, temperature=temp, model_name=model)
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )
        return conversation_chain

    def get_url_text(url):
        headers = {'Accept-Encoding': 'identity'}
        responseData = requests.get(url, headers=headers)
        content = responseData.text
        
        SaveData.save_attr(url)
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()
        for script in soup(["script", "style"]):
            script.extract()

        cleaned_text_content = soup.get_text()
        return str(text)
    
    
    # Single PDF
    def get_pdf_text(pdf):
        SaveData.save_attr(pdf)
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    # Multiple PDFs
    def get_pdfs_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            text_temp = ""
            SaveData.save_attr(pdf)
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text_temp += page.extract_text()
            text += text_temp
        return text
    
    # Single Txt
    def get_txt_text(text_file):
        text = ""
        SaveData.save_attr(text_file)
        for line in text_file:
                text = text +  str(line.decode())
        return text

    # Multiple Txts
    def get_txts_text(txt_files):
        text = ""
        for txt in txt_files:
            SaveData.save_attr(txt)
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
        raw_text = LibForEmbedding.get_url_text(url)
        text_chunks = LibForEmbedding.get_text_chunks(raw_text)
        file_id = LangChainAttr.objects.latest('id').id
        vectorstore = LibForEmbedding.get_vectorstore(text_chunks, file_id)

        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)    
    
    
class EmbeddingPDF(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        files = request.FILES.getlist('files')

        for file in files:
            raw_text = LibForEmbedding.get_pdf_text(file)
            text_chunks = LibForEmbedding.get_text_chunks(raw_text)
            file_id = LangChainAttr.objects.latest('id').id
            vectorstore = LibForEmbedding.get_vectorstore(text_chunks, file_id)
        
        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)


class EmbeddingCSV(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        input_data = request.data
        files = request.FILES.getlist('files')

        if len(files) == 0:
            return Response({"status": 'error', "data": 'validate error' }, status=status.HTTP_201_CREATED)    

        for csv_file in files:
            EMBEDDING_VAL = files   
            EMBEDDING_TYPE = 'csv'
            # Load CSV data into Langchain
            file_path = 'tmp/temp_uploaded_file.csv'
            with open(file_path, 'wb+') as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)
            destination.close()

            csv_loader = CSVLoader(
                'tmp/temp_uploaded_file.csv',
                encoding="utf-8",
                csv_args={
                    # 'quotechar': '"',
                    'fieldnames': ['OrderDate', 'Region', 'Rep', 'Item', 'Sold Units', 'Unit Cost', 'Total'],
                    'delimiter': ','
                },
            )
            documents = csv_loader.load()
            print("documents = ", documents)
            text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=0)
            text_chunks = text_splitter.split_documents(documents)
            # vectorstore = LibForEmbedding.get_vectorstore(text_chunks,csv_file)
            vectorstore = LibForEmbedding.get_vectorstore(documents, files)

            SaveData.save_attr(csv_file)
        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)    


class EmbeddingTXT(APIView):
    renderer_classes = [UserRenderer]

    def post(self, request, format=None):
        files = request.FILES.getlist('files')

        for file in files:
            # request.session["conversation"] = None
            raw_text = LibForEmbedding.get_txt_text(file)
            text_chunks = LibForEmbedding.get_text_chunks(raw_text)
            file_id = LangChainAttr.objects.latest('id').id
            vectorstore = LibForEmbedding.get_vectorstore(text_chunks, file_id)
        
        return Response({"status": 'success', "data": 'success' }, status=status.HTTP_201_CREATED)    
    

class CHAT(APIView):
    
    def post(self, request, format=None):
        input_data = request.data
        user_promts = input_data['userPrompts']
        prompt = set_prompt()
        history = []
        fileList = input_data['fileList']
        
        if 'history' in input_data:
            history = input_data['history']
        MODEL = input_data['modal']
        
        print("modal = ", MODEL)
        current_date = datetime.now().date()
        desired_difference = current_date - timedelta(days=7)
        # .filter(Q(attribute__in=fileList))    
        approve_file_list = LangChainAttr.objects.annotate(date_difference=F('created_at') - desired_difference).filter(date_difference__lte=timedelta(days=7))
        # results = LangChainAttr.objects.annotate(date_difference=F('created_at') - desired_difference).filter(date_difference__lte=timedelta(days=7))
        if len(approve_file_list) == 0:
            return Response({"status": 'success', "answer": "expired embedding data.", "history" :history }, status=status.HTTP_201_CREATED)    
        embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key.attribute)
        
        # PineCone
        # index = pinecone.Index(PINECONE_INDEX_NAME)
        # vectorstore = Pinecone(index, embedding.embed_query, "text")
        
        # FAISS
        vectorstore = FAISS.load_local(f"./store/", embeddings, fileList[0])
        for file in fileList:
            if(file == fileList[0]): continue
            loaded_vectorestore = FAISS.load_local(f"./store/", embeddings, file)
            vectorstore.merge_from(loaded_vectorestore)
            
        conversation = LibForEmbedding.get_conversation_chain(
            vectorstore,
            temp=TEMP,
            model=MODEL
        )
        temp = [tuple((row[0], row[1]) for row in history)]
        conversation_result = conversation(
            {
                'question': (prompt+user_promts), 
                'chat_history': temp
            }
        )
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
        prompt = set_prompt()
        history = []
        MODEL = 'gpt-3.5-turbo'
        stripped_user_promps = text.strip()
        index = pinecone.Index(PINECONE_INDEX_NAME)
        embedding = OpenAIEmbeddings(openai_api_key=open_ai_key.attribute)
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
        server = os.environ["EMAIL_SERVER"]
        EMAIL_ADDRESS = os.environ["USER_MAIL_ADDRESS"]
        EMAIL_PASSWORD = os.environ["USER_MAIL_PASSWORD"]
        with imaplib.IMAP4_SSL('imap.gmail.com') as mail:
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            mail.select('inbox')

            _, data = mail.search(None, 'UNSEEN')
            for num in data[0].split():
                _, msg_data = mail.fetch(num, '(RFC822)')

                raw_email = msg_data[0][1]
                msg = EmailMessage()
                msg = Parser().parsestr(raw_email.decode())

                subject = msg['Subject']
                sender = msg['From']
                messages = msg.as_string()
                msg = EmailMessage()
                sanitized_subject = sanitize_header(subject)
                sanitized_recipient = sanitize_header(sender)
                msg['Subject'] = f'Re: {sanitized_subject}'
                msg['From'] = EMAIL_ADDRESS
                msg['To'] = sanitized_recipient
                msg.set_content('Thank you for your email. This is an automated response.')
                with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                    smtp.starttls()
                    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    smtp.send_message(msg)
        return Response({"status": "success", "message": "save successfully"}, status=status.HTTP_201_CREATED)  
    
class GetEmbeddingData(APIView):
    def post(self, request, format=None):
        current_date = datetime.now().date()
        desired_difference = current_date - timedelta(days=7)
        file_list = LangChainAttr.objects.annotate(date_difference=F('created_at') - desired_difference).filter(Q(date_difference__lte=timedelta(days=7)) & Q(attr_type='document'))
        result = []
        for row in file_list:
            result.append(
                [
                    row.id,
                    row.attribute
                ]
            )
        return Response(data = result,status=status.HTTP_200_OK)