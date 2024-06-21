from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

import json 
import os 
import sys
import boto3

bedrock=boto3.client(service_name="bedrock-runtime")
bedrock_embedding=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    loader=PyPDFDirectoryLoader("./data")
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs=text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vector_store=FAISS.from_documents(docs, bedrock_embedding)
    vector_store.save_local("faiss_index")
    return vector_store


if __name__=="__main__":
    docs=data_ingestion()
    get_vector_store(docs)
