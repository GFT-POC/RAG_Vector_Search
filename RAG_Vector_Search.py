import streamlit as st
import pdfplumber
import os
from groq import Groq
from transformers import AutoModel, AutoTokenizer
#from pinecone import Pinecone, ServerlessSpec
import pinecone
#from pinecone import ServerlessSpec
import unicodedata

from transformers import AutoModel

# Load API keys securely from environment variables (You should set these environment variables securely)
API_KEY = "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"
PINECONE_API_KEY = "2877c19f-9191-4257-8d04-c149c8edb028"
PINECONE_ENV = "us-west1-gcp" #"us-east1-gcp"

if not API_KEY or not PINECONE_API_KEY:
    st.error("API keys are missing. Please set the GROQ_API_KEY and PINECONE_API_KEY environment variables.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Initialize Pinecone client using the new API
try:
    pc = pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    #pc = pinecone(api_key=PINECONE_API_KEY)
    #st.write("Pinecone initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    st.stop()

# Load embedding model for RAG
embedding_model = AutoModel.from_pretrained('bert-base-uncased') #AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')

# Title and Short Description
st.title("Intelligent Document Search App")
st.write("This app allows you to search and retrieve relevant information from documents stored in a vector database. "
         "You can choose between searching for candidate resumes or performing a general search to find the most relevant content based on your query.")

# Initialize Streamlit session state for vector store selection
if 'selected_vector_store' not in st.session_state:
    st.session_state.selected_vector_store = None

# Dropdown to select existing vector stores or create a new one
st.header("Select or Create Vector Store")

# Get a list of existing vector stores
existing_indexes = pc.list_indexes().names()

# Initialize a placeholder for `index_name`
index_name = None

if existing_indexes:
    vector_store_choice = st.selectbox(
        "Choose a vector store or create a new one:",
        options=["Create New"] + existing_indexes
    )
else:
    vector_store_choice = "Create New"
    st.write("No existing vector stores found, please create a new one.")

# If user chooses to create a new vector store
if vector_store_choice == "Create New":
    new_vector_store_name = st.text_input("Enter the name for the new vector store")
    if new_vector_store_name:
        index_name = new_vector_store_name
        # Create a new index with the correct dimension (768 in this case)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,  # Ensure this matches your embedding model's dimension
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east1')
            )
            st.success(f"New vector store '{index_name}' created.")
else:
    index_name = vector_store_choice

# Reset the session state when the vector store is changed
if st.session_state.selected_vector_store != vector_store_choice:
    # Reset relevant session state variables
    st.session_state.clear()
    st.session_state.selected_vector_store = vector_store_choice
    st.experimental_rerun()  # Force a rerun of the script to "reinitialize" the interface

# Ensure that index_name is defined before creating the index
if index_name:
    try:
        index = pc.Index(index_name)
    except Exception as e:
        st.error(f"Error connecting to Pinecone index: {e}")
        st.stop()
else:
    st.error("Please select or create a vector store before uploading files.")
    st.stop()

# Function to normalize file names
def normalize_filename(filename):
    normalized_filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    return normalized_filename

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Embedding generation
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings[0].tolist()

# Define query_database function to retrieve context from Pinecone based on a query
def query_database(query, top_k_chunks_per_file=3, top_k_files=5, total_chunk_limit=15):
    query_embedding = get_embeddings(query)
    try:
        result = index.query(vector=query_embedding, top_k=100, include_metadata=True)
        if result['matches']:
            file_chunks = {}
            for match in result['matches']:
                file_name = match['metadata']['file_name']
                chunk_text = match['metadata']['chunk_text']
                score = match['score']

                if file_name not in file_chunks:
                    file_chunks[file_name] = []
                file_chunks[file_name].append((chunk_text, score))

            top_file_chunks = {}
            for file_name, chunks in file_chunks.items():
                top_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)[:top_k_chunks_per_file]
                top_file_chunks[file_name] = top_chunks

            top_files = sorted(top_file_chunks.items(), key=lambda x: x[1][0][1], reverse=True)[:top_k_files]

            selected_chunks = []
            for file_name, chunks in top_files:
                for chunk, score in chunks:
                    selected_chunks.append((file_name, chunk, score))

            selected_chunks = sorted(selected_chunks, key=lambda x: x[2], reverse=True)[:total_chunk_limit]

            context = ""
            for file_name, chunk_text, score in selected_chunks:
                context += f"File: {file_name}\nScore: {score:.4f}\n\nText: {chunk_text}\n\n"

            return context if context else "No relevant matches found in the database."
        else:
            return "No relevant matches found in the database."
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return ""

# Define call_groq function to handle the question-answering process using Groq
def call_groq(user_query, context, search_type):
    # Modify the system prompt based on the selected search type
    if search_type == "Candidate Resume Search":
        system_prompt = f"""
            You will be provided with several chunks of text extracted from resumes of potential candidates.\
            These chunks have been identified through an embedding vector search based on the user query.\

            Your task is to:\

            Identify the names of potential candidates mentioned in these text chunks who best meet the \
            requirements described in the user query.\
            
            Provide an explanation for why each candidate meets the criteria. Use details from the text \
            such as skills, experience, education, certifications, or specific project accomplishments \
            that align with the user requirements.\
            
            Candidate Name: [Name]\
            Explanation: [A brief explanation of how the candidate fulfills the query requirements]\
            
            Here's the information to analyze:\

            Chunks: {context}\
            User question: {user_query}\
        """
    else:
        system_prompt = f"""
        Instructions:\
        - You are performing a general search. Be helpful and answer questions concisely.\
        - Use the provided file names and text chunks to answer the question accurately.\
        Context: {context}\
        User question: {user_query}\
        """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": system_prompt}],
            model="LLaMA3-70b-8192",
            temperature=0.8,
            max_tokens=8000,
            top_p=1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return "There was an error processing your request."

# Display files in the selected vector store only after a vector store is selected
if vector_store_choice != "Create New" and index_name:
    st.header(f"Files in {index_name}")
    
    try:
        # Use a query to retrieve all vectors, retrieving metadata
        vectors_info = index.query(vector=[0.0]*768, top_k=10000, include_metadata=True)
        
        # Extract unique file names from metadata and store vector IDs for each file
        if 'matches' in vectors_info and vectors_info['matches']:
            files_in_store = set()
            vector_ids_per_file_prefix = {}
            
            for match in vectors_info['matches']:
                full_file_name = match['metadata'].get('file_name')
                if full_file_name:
                    files_in_store.add(full_file_name)
                    if full_file_name not in vector_ids_per_file_prefix:
                        vector_ids_per_file_prefix[full_file_name] = []
                    vector_ids_per_file_prefix[full_file_name].append(match['id'])  # Store the vector ID for deletion
            
            if files_in_store:
                files_to_remove = st.multiselect("Select files to remove:", list(files_in_store))
                
                if st.button("Remove Selected Files"):
                    vector_ids_to_delete = []
                    for file in files_to_remove:
                        for full_file_name in vector_ids_per_file_prefix:
                            if full_file_name.startswith(file):
                                vector_ids_to_delete.extend(vector_ids_per_file_prefix.get(full_file_name, []))
                    
                    if vector_ids_to_delete:
                        index.delete(ids=vector_ids_to_delete)
                        st.success(f"Removed all chunks of '{', '.join(files_to_remove)}' from vector store.")
                    else:
                        st.warning("No vectors found to delete for the selected files.")
            else:
                st.write("No files found in the selected vector store.")

        else:
            st.write("No files found in the selected vector store.")

        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"], key="file_uploader_unique_key")

        def file_exists_as_prefix(uploaded_file_name):
            for full_file_name in files_in_store:
                if full_file_name.startswith(uploaded_file_name):
                    return True
            return False

        if uploaded_files:
            for uploaded_file in uploaded_files:
                normalized_filename = normalize_filename(uploaded_file.name)
                text = extract_text_from_pdf(uploaded_file)

                if not text:
                    st.error(f"Could not extract text from {uploaded_file.name}.")
                    continue

                if file_exists_as_prefix(normalized_filename):
                    replace_file = st.radio(
                        f"A file with the prefix '{normalized_filename}' already exists. Do you want to replace it?",
                        options=["Yes, replace", "No, keep existing data"],
                        index=1,
                        key=f"replace_radio_{normalized_filename}"
                    )

                    if replace_file == "Yes, replace":
                        vector_ids_to_delete = []
                        for full_file_name in vector_ids_per_file_prefix:
                            if full_file_name.startswith(normalized_filename):
                                vector_ids_to_delete.extend(vector_ids_per_file_prefix.get(full_file_name, []))
                        
                        if vector_ids_to_delete:
                            index.delete(ids=vector_ids_to_delete)
                            st.success(f"Removed all chunks of '{normalized_filename}' from vector store.")

                chunks = [text[i:i + 200] for i in range(0, len(text), 200)]
                for i, chunk in enumerate(chunks):
                    embedding = get_embeddings(chunk)
                    vector_id = f"{normalized_filename}_{i}"
                    index.upsert(vectors=[(vector_id, embedding, {"chunk_text": chunk, "file_name": uploaded_file.name})])
                
                st.success(f"Embeddings for '{uploaded_file.name}' added to the vector store '{index_name}'.")

    except Exception as e:
        st.error(f"Error fetching vector store data: {e}")

# Search Type Dropdown
search_type = st.selectbox("Select Search Type:", options=["Candidate Resume Search", "General Search"], index=0)

# Search interface
user_query = st.text_input("Enter your search query")

if user_query:
    context = query_database(user_query)
    if context:
        response = call_groq(user_query, context, search_type)
        st.write(response)
    else:
        st.warning("No relevant context found for the query.")
