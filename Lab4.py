import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from PyPDF2 import PdfReader

# A fix for working with ChromaDB on Streamlit Community Cloud
# This MUST run before chromadb is imported
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Create ChromaDB client
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection(name="Lab4Collection")

### Using Chroma DB with OpenAI embeddings ###

# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# A function that will add documents to collection
# collection = ChromaDB collection, already established
# text = extracted text from PDF files
# Embeddings inserted into the collection from OpenAI
def add_documents_to_collection(collection, text, file_name):

    #Create an embedding
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    #Create an embedding
    embedding = response.data[0].embedding

    # Add the document and its embedding to ChromaDB
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding]
    )

#### Extract text from PDF files ####
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + '\n'
    return text.strip()

#### POPULATE COLLECTION WITH PDFs ####
# This function uses extract_text_from_pdf
# and add_documents_to_collection to put syllabi in ChromaDB collection
def load_pdfs_to_collection(folder_path, collection):
    folder = Path(folder_path)

    if not folder.is_dir():
        st.error(f'Could not find the folder {folder_path}')
        return 0

    pdf_files = sorted(folder.glob('*.pdf'))
    loaded = 0
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if not text:
            st.warning(f'No text extracted from {pdf_file.name} - skipping')
            continue
        add_documents_to_collection(collection, text, pdf_file.name)
        loaded += 1
    return loaded

# Check if collection is empty and load PDFs
if collection.count() == 0:
    loaded = load_pdfs_to_collection('./Lab-04-Data/', collection)

#### Store the vector database collection in st.session_state.Lab4_VectorDB
if 'Lab4_VectorDB' not in st.session_state:
    st.session_state.Lab4_VectorDB = collection

#### MAIN APP ####
st.title('Lab 4: Chatbot using RAG')

#### GET RELEVANT INFO FROM THE VECTOR DB ####
# Embeds the user's question and returns the 3 closest syllabi
def get_info_from_vectordb(collection, query):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small')

    # Get the embedding
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # Build one block of text out of the returned syllabi
    documents = results['documents'][0]
    ids = results['ids'][0]

    extra_info = ''
    for doc_id, doc in zip(ids, documents):
        extra_info += f'--- SYLLABUS: {doc_id} ---\n'
        extra_info += doc[:6000] + '\n\n'

    return extra_info, ids


#### THE CHATBOT ####
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 'content': 'Ask me anything about the iSchool courses.'}
    ]

# Display chat messages from history on app rerun
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg['role'])
    chat_msg.write(msg['content'])

# React to user input
if prompt := st.chat_input('What would you like to know?'):

    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    # Get the relevant syllabi for this question
    extra_info, source_ids = get_info_from_vectordb(collection, prompt)

    system_prompt = (
        'You are a course information assistant for the Syracuse iSchool. '
        'Use the course syllabi below to answer the question.\n\n'
        'Rules:\n'
        '- When you use the syllabi, start your answer with "Based on the course syllabi:" '
        'and name the file(s) you used.\n'
        '- If the syllabi do not answer the question, say so, then start with '
        '"Answering from general knowledge:" before continuing.\n'
        '- Do not make up course numbers, instructors, or dates.\n\n'
        'Course syllabi:\n' + extra_info
    )

    # Keep the last 6 messages as conversation memory
    buffer = st.session_state.messages[-6:]
    while buffer and buffer[0]['role'] == 'assistant':
        buffer = buffer[1:]

    # Get LLM response
    with st.chat_message('assistant'):
        client = st.session_state.openai_client
        stream = client.chat.completions.create(
            model='gpt-5-mini',
            messages=[{'role': 'system', 'content': system_prompt}] + buffer,
            stream=True,
        )
        response = st.write_stream(stream)

    # Add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    