import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import numpy as np
import PyPDF2
import faiss
import os
import pickle
from pathlib import Path

# Configuration
mistral_api_key = "6mEENhDapV7fw5uVvvUvenytRVJ23ghV"  # Replace with your actual API key
knowledge_base_dir = "knowledge_base"  # Directory to store PDFs
embeddings_cache_file = "embeddings_cache.pkl"  # File to store embeddings
MODEL_NAME = "open-mistral-7b"  # Model to use for chat
EMBED_MODEL = "mistral-embed"  # Model to use for embeddings
CHUNK_SIZE = 4096  # Size of text chunks
TOP_K = 4  # Number of chunks to retrieve

# Initialize Mistral client
cli = MistralClient(api_key=mistral_api_key)

# Global variables for knowledge base
pdf_chunks = []
embeddings_index = None

def setup_knowledge_base():
    """Set up the knowledge base by loading PDFs and creating embeddings"""
    global pdf_chunks, embeddings_index
    
    # Create knowledge base directory if it doesn't exist
    if not os.path.exists(knowledge_base_dir):
        os.makedirs(knowledge_base_dir)
        print(f"Created knowledge base directory at {knowledge_base_dir}")
        print("Please add your PDF files to this directory and restart the application")
        return False
    
    # Check if we have PDFs in the knowledge base
    pdf_files = list(Path(knowledge_base_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {knowledge_base_dir}")
        print("Please add your PDF files to this directory and restart the application")
        return False
    
    # Try to load cached embeddings
    if os.path.exists(embeddings_cache_file):
        try:
            with open(embeddings_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                pdf_chunks = cache_data['chunks']
                embeddings = cache_data['embeddings']
                
                # Create FAISS index from cached embeddings
                d = embeddings.shape[1]
                embeddings_index = faiss.IndexFlatL2(d)
                embeddings_index.add(embeddings)
                
                print(f"Loaded {len(pdf_chunks)} chunks from cache")
                return True
        except Exception as e:
            print(f"Error loading cache: {e}")
            # Continue with rebuilding the cache
    
    # Process PDFs and create chunks
    pdf_chunks = []
    print(f"Processing {len(pdf_files)} PDF files...")
    
    for pdf_path in pdf_files:
        try:
            print(f"Reading {pdf_path.name}...")
            reader = PyPDF2.PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text()
            
            # Split text into chunks
            chunks = [pdf_text[i:i + CHUNK_SIZE] for i in range(0, len(pdf_text), CHUNK_SIZE)]
            pdf_chunks.extend(chunks)
            print(f"Added {len(chunks)} chunks from {pdf_path.name}")
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    
    if not pdf_chunks:
        print("No text could be extracted from the PDFs")
        return False
    
    # Generate embeddings for all chunks
    print(f"Generating embeddings for {len(pdf_chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(pdf_chunks):
        print(f"Embedding chunk {i+1}/{len(pdf_chunks)}...")
        embedding = get_text_embedding(chunk)
        embeddings.append(embedding)
    
    # Convert to numpy array and create FAISS index
    embeddings = np.array(embeddings)
    d = embeddings.shape[1]
    embeddings_index = faiss.IndexFlatL2(d)
    embeddings_index.add(embeddings)
    
    # Cache the embeddings for faster startup next time
    with open(embeddings_cache_file, 'wb') as f:
        pickle.dump({'chunks': pdf_chunks, 'embeddings': embeddings}, f)
    
    print(f"Knowledge base created with {len(pdf_chunks)} chunks")
    return True

def get_text_embedding(input: str):
    """Get embedding for a text using Mistral API"""
    embeddings_batch_response = cli.embeddings(
        model=EMBED_MODEL,
        input=input
    )
    return embeddings_batch_response.data[0].embedding

def retrieve_relevant_chunks(question: str) -> str:
    """Retrieve the most relevant chunks for a question"""
    global pdf_chunks, embeddings_index
    
    if not pdf_chunks or embeddings_index is None:
        return "Knowledge base not loaded. Please restart the application."
    
    # Get embedding for the question
    question_embedding = np.array([get_text_embedding(question)])
    
    # Search for similar chunks
    D, I = embeddings_index.search(question_embedding, k=min(TOP_K, len(pdf_chunks)))
    retrieved_chunks = [pdf_chunks[i] for i in I.tolist()[0]]
    
    # Join the chunks
    text_retrieved = "\n\n".join(retrieved_chunks)
    return text_retrieved

def ask_mistral(message: str, history: list):
    print(f"Received message: {message}")  # Debugging line
    messages = []

    # Convert history to ChatMessage format
    for msg in history:
        # Check if history item is a dictionary (type="messages" case)
        if isinstance(msg, dict):
            messages.append(ChatMessage(role=msg.get("role", "user"), content=msg.get("content", "")))
        # Otherwise, assume tuple format (type="chat" case)
        else:
            messages.append(ChatMessage(role="user", content=msg[0]))
            messages.append(ChatMessage(role="assistant", content=msg[1]))

    # Get relevant chunks from knowledge base
    retrieved_text = retrieve_relevant_chunks(message)

    # Create prompt with retrieved context
    prompt = (
        f"Context from knowledge base:\n{retrieved_text}\n\n"
        f"Question: {message}\n\n"
        "Please answer based on the context provided."
    )
    messages.append(ChatMessage(role="user", content=prompt))

    # Stream the response
    full_response = ""
    try:
        for chunk in cli.chat_stream(model=MODEL_NAME, messages=messages, max_tokens=1024):
            full_response += chunk.choices[0].delta.content
    except Exception as e:
        print(f"Error during chat stream: {e}")  # Log any errors
        return "An error occurred while generating a response."

    return full_response

def add_pdf(files):
    """Add new PDFs to the knowledge base"""
    if not files:
        return "No files provided"
    
    # Save files to knowledge base directory
    count = 0
    for file in files:
        if file.name.lower().endswith('.pdf'):
            target_path = os.path.join(knowledge_base_dir, file.name)
            with open(target_path, 'wb') as f:
                f.write(file.getvalue())  # Use getvalue() to get the content of NamedString
            count += 1
    
    if count == 0:
        return "No PDF files were provided"
    
    # Delete cache to force regeneration
    if os.path.exists(embeddings_cache_file):
        os.remove(embeddings_cache_file)
    
    return f"Added {count} PDFs to knowledge base. Please restart the application to update the knowledge base."

def main():
    """Main function to set up the Gradio interface"""
    # Set up knowledge base
    kb_ready = setup_knowledge_base()
    
    # Create Gradio interface
    with gr.Blocks(title="PDF Knowledge Base Chat") as app:
        gr.Markdown("# Chat with your PDF Knowledge Base")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=ask_mistral,
                    title="",
                    chatbot=gr.Chatbot(height=600),
                    textbox=gr.Textbox(placeholder="Ask a question about your PDFs...", container=False),
                    submit_btn="Send",
                    type='messages'
                )
            
            with gr.Column(scale=1):
                gr.Markdown("## Add PDFs to Knowledge Base")
                file_output = gr.Textbox(label="Upload Status")
                file_input = gr.File(
                    file_count="multiple",
                    label="Upload PDFs",
                    file_types=[".pdf"],
                )
                file_input.upload(add_pdf, file_input, file_output)
                
                gr.Markdown("### Knowledge Base Status")
                if kb_ready:
                    status = f"✅ Knowledge base loaded with {len(pdf_chunks)} text chunks"
                else:
                    status = "❌ Knowledge base not loaded"
                gr.Markdown(status)
                
                gr.Markdown("""
                ### Instructions
                1. Upload PDF files using the panel on the right
                2. Restart the application to update the knowledge base
                3. Ask questions in the chat interface
                
                All PDFs are stored in the `knowledge_base` directory.
                """)
    
    app.launch()

if __name__ == "__main__":
    main()
