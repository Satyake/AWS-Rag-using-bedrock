from langchain_community .embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock  
from langchain.prompts import PromptTemplate 
from langchain.chains import RetrievalQA 
from langchain.vectorstores import FAISS
from ingestion import data_ingestion, get_vector_store 
from retrieval import get_llama2_llm,get_response_llm
import boto3
import streamlit as st

bedrock=boto3.client(service_name='bedrock-runtime')
bedrock_embedding=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", 
client=bedrock)

def main():
    st.set_page_config("QA with Doc")
    st.header("QA with doc")

    user_question=st.text_input("Ask question from pdf file")

    with st.sidebar:
        st.title("update or create the vectorstore")
        if st.button("vectors_update"):
            with st.spinner('processing'):
                docs=data_ingestion()
                get_vector_store(docs) 
                st.success("Done")
        if st.button("llama model"):
            with st.spinner('processing'):
                faiss_index=FAISS.load_local("faiss_index", bedrock_embedding, allow_dangerous_deserialization=True)
                llm=get_llama2_llm()
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")

if __name__=='__main__':
    main()