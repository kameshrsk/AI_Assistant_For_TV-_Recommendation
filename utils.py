from langchain_community.document_loaders import WebBaseLoader
import bs4
import re
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


urls=[
    "https://www.rtings.com/tv/reviews/lg/g4-oled", "https://www.rtings.com/tv/reviews/sony/a95l-oled", "https://www.rtings.com/tv/reviews/lg/g3-oled",
    "https://www.rtings.com/tv/reviews/sony/a95k-oled", "https://www.rtings.com/tv/reviews/samsung/s95c-oled", "https://www.rtings.com/tv/reviews/samsung/s95d-oled",
    "https://www.rtings.com/tv/reviews/lg/c4-oled", "https://www.rtings.com/tv/reviews/samsung/s90d-s90dd-oled-qd-oled","https://www.rtings.com/tv/reviews/samsung/s90c-oled",
    "https://www.rtings.com/tv/reviews/lg/c3-oled", "https://www.rtings.com/tv/reviews/sony/bravia-8-oled", "https://www.rtings.com/tv/reviews/lg/b4-oled",
    "https://www.rtings.com/tv/reviews/samsung/s89c-oled", "https://www.rtings.com/tv/reviews/sony/a90j-oled", "https://www.rtings.com/tv/reviews/lg/g2-oled"
]

loader=WebBaseLoader(
    web_path=urls,
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=["e-rich_content","product_page-summary", "product_page-body", "e-simple_grid is-aligned"]
    ))
)

docs=loader.load()

def clean_text(text):
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespaces
    text = text.strip()
    return text

for i in range(len(docs)):

    docs[i].page_content=clean_text(docs[i].page_content)

load_dotenv("UnBoxing/.env")

splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

splitted_docs=splitter.split_documents(docs)

vector_db=FAISS.from_documents(splitted_docs, HuggingFaceEmbeddings())

retriever=vector_db.as_retriever()