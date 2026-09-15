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

#### QUERYING A COLLECTION -- ONLY USED FOR TESTING ####
topic = st.sidebar.text_input('Topic', placeholder='Type your topic (e.g., GenAI)...')

if topic:
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=topic,
        model='text-embedding-3-small')

    # Get the embedding
    query_embedding = response.data[0].embedding

    # Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # The number of closest documents to return
    )

    # Display the results
    st.subheader(f'Results for: {topic}')

    for i in range(len(results['documents'][0])):
        doc_id = results['ids'][0][i]
        st.write(f'**{i+1}. {doc_id}**')

else:
    st.info('Enter a topic in the sidebar to search the collection')

# if "messages" not in st.session_state:
#     st.session_state.messages = \
#     [{"role": "assistant", "content": "How can I help you?"}]

# #Display chat messages from history on app rerun
# for msg in st.session_state.messages:
#     chat_msg = st.chat_message(msg["role"])
#     chat_msg.write(msg["content"])

# # React to user input
# if prompt := st.chat_input("What is up?"):

#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     buffer = st.session_state.messages[-6:]
#     while buffer and buffer[0]["role"] == "assistant":
#         buffer = buffer[1:]

#     with st.chat_message("assistant"):
#         try:
#             if vendor == "openai":
#                 stream = stream_openai(model_name, SYSTEM_PROMPT, buffer)
#             else:
#                 stream = stream_gemini(model_name, SYSTEM_PROMPT, buffer)
#             response = st.write_stream(stream)
#         except Exception as e:
#             response = f"Sorry, something went wrong: {e}"
#             st.error(response)

#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})