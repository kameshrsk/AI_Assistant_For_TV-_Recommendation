from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import streamlit as st
from streamlit_chat import message
from utils import *
import uuid
import re

load_dotenv(dotenv_path='/teamspace/studios/this_studio/UnBoxing/.env')

os.environ['MISTRAL_API_KEY']=os.getenv('MISTRAL_API_KEY')
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

template = '''
You are an AI assistant chatbot that helps customers recommend a TV based on their needs. Follow these guidelines:

Gather essential information about the user's needs and preferences.
Ask about primary purpose, budget, size preferences, and specific features they're looking for.
Do not make specific TV recommendations until you have gathered at least 3 key preferences.
When you have sufficient information, base your recommendations on the following:
User Preferences:
{user_preferences}

Retrieved Information:
{context}

Chat History:
{chat_history}

Current Human Query: {question}

Remember to:
- Be patient and thorough in gathering information.
- Avoid repeating questions already answered.
- Explain why new brands might be good alternatives based on the user's preferences.
- If the user asks for recommendations before you have enough information, politely ask for more details.
- If the user indicates they're done or satisfied, politely conclude the conversation.

Chatbot:
'''


prompt=PromptTemplate(
    input_variables=['user_preferences', 'context', 'chat_history', 'question'],
    template=template
)

llm=ChatGroq(
    model='llama-3.1-70b-versatile',
    groq_api_key=os.environ['GROQ_API_KEY']
)

def is_termination_statement(text):
    termination_phrases = [
        r"that's fine for now",
        r"i'll decide later",
        r"that's enough",
        r"thank you for your help",
        r"i'll think about it",
        r"i'll go over the options",
        r"that's all I need"
    ]
    return any(re.search(phrase, text, re.IGNORECASE) for phrase in termination_phrases)

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state['session_id']=str(uuid.uuid4())

    return st.session_state['session_id']

def get_session_memory(session_id):
    if 'memories' not in st.session_state:
        st.session_state['memories']={}

    if session_id not in st.session_state['memories']:
        st.session_state['memories'][session_id]=ConversationBufferMemory(
            memory_key='chat_history',
            input_key='question',
            return_messages=True
        )

    return st.session_state['memories'][session_id]

def extract_preferences(text):
    preferences = {}
    
    # Extract primary purpose
    purpose_match = re.search(r'(?i)used.*for\s+(\w+)', text)
    if purpose_match:
        preferences['primary_purpose'] = purpose_match.group(1)
    
    # Extract budget
    budget_match = re.search(r'(?i)budget.*?(\d+)', text)
    if budget_match:
        preferences['budget'] = int(budget_match.group(1))
    
    # Extract size preference
    size_match = re.search(r'(?i)(\d+)\s*inch', text)
    if size_match:
        preferences['size'] = int(size_match.group(1))
    
    # Extract brand preference
    brand_match = re.search(r'(?i)prefer\s+(\w+)', text)
    if brand_match:
        preferences['brand_preference'] = brand_match.group(1)
    
    # Add more extractions as needed
    
    return preferences

def get_session_preferences(session_id):
    if 'preferences' not in st.session_state:
        st.session_state['preferences']={}

    if session_id not in st.session_state['preferences']:
        st.session_state['preferences'][session_id]={}

    return st.session_state['preferences'][session_id]

def get_session_chain(session_id):
    memory=get_session_memory(session_id)

    if 'chains' not in st.session_state:
        st.session_state['chains']={}

    if session_id not in st.session_state['chains']:
        st.session_state['chains'][session_id]=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_type='mmr', search_kwargs={"k":10}),
        memory=memory,
        combine_docs_chain_kwargs={'prompt':prompt}
    )

    return st.session_state['chains'][session_id]

def update_user_preferences(session_id, question, answer):
    preferences = get_session_preferences(session_id)
    
    # Extract preferences from both question and answer
    new_prefs_from_question = extract_preferences(question)
    new_prefs_from_answer = extract_preferences(answer)
    
    # Update preferences
    preferences.update(new_prefs_from_question)
    preferences.update(new_prefs_from_answer)

def get_chain_response(session_id, question):
    chain = get_session_chain(session_id)
    user_prefs = get_session_preferences(session_id)
    user_prefs_str = "\n".join([f"{k}: {v}" for k, v in user_prefs.items()])
    
    if is_termination_statement(question):
        return "I understand you're done for now. Thank you for using our TV recommendation service. If you need any more information in the future, please don't hesitate to ask. Have a great day!"
    
    output = chain.invoke({
        "question": question,
        "user_preferences": user_prefs_str
    })
    return output['answer']

st.title("The UnBoxing Assistant")

session_id = get_session_id()

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'responses' not in st.session_state:
    st.session_state['responses'] = ['Hi. How may I help you with TV recommendations today?']

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

response_container = st.container()
container = st.container()

with container:
    question = st.text_input("Query:")

    if question:
        with st.spinner("Processing..."):
            response_text = get_chain_response(session_id, question)
            update_user_preferences(session_id, question, response_text)

            st.session_state['responses'].append(response_text)
            st.session_state['requests'].append(question)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user=True, key=str(i)+"_user")