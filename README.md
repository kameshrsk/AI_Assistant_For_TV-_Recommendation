# TV Recommendation Chatbot

This project implements an AI-powered chatbot that provides TV recommendations based on user preferences. It uses a combination of web scraping, document processing, and conversational AI to deliver personalized suggestions.

## Project Structure

- `utils.py`: Handles data collection, processing, and vector database creation.
- `app_with_session_management.py`: Main application file for the Streamlit-based chatbot interface.
- `requirements.txt`: Lists all required Python packages.
- `.env`: Contains API keys and other environment variables (not included in the repository).

## Setup and Installation

1. Clone the repository:

    git clone https://github.com/kameshrsk/AI_Assistant_For_TV-_Recommendation
    cd <repository-directory>

2. Create a virtual environment and activate it

3. Install the required packages:

    pip install -r requirements.txt

4. Create a `.env` file in the project root and add your API keys:

    MISTRAL_API_KEY=your_mistral_api_key
    GROQ_API_KEY=your_groq_api_key

## Usage

1. First, run the `utils.py` script to create and save the vector database:

    python utils.py

    This script will:
    - Scrape TV review data from specified URLs
    - Process and clean the text data
    - Create a FAISS vector database
    - Save the database locally

2. Once the vector database is created, run the Streamlit app:

    streamlit run app_with_session_management.py

    This will start the chatbot interface in your default web browser.

3. Interact with the chatbot by entering your TV preferences and requirements. The chatbot will use the vector database to retrieve relevant information and provide personalized recommendations.

## How It Works

1. Data Collection and Processing (`utils.py`):
- Web scraping using `WebBaseLoader`
- Text cleaning and processing
- Document splitting using `RecursiveCharacterTextSplitter`
- Vector database creation using FAISS and HuggingFace embeddings

2. Chatbot Interface (`app_with_session_management.py`):
- Streamlit-based user interface
- Session management for handling multiple users
- Conversation history tracking
- Integration with Groq LLM for natural language processing
- Retrieval of relevant information from the vector database

3. Recommendation Process:
- The chatbot gathers user preferences through conversation
- It uses these preferences to query the vector database
- Relevant TV information is retrieved and processed by the LLM
- The LLM generates personalized recommendations based on the user's needs and the retrieved data

