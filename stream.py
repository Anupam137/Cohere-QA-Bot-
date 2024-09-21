import os
import pinecone
import cohere
import torch
import PyPDF2
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from io import BytesIO

# Set up Pinecone and Cohere
PINECONE_API_KEY = "Your pinecone api Key"
COHERE_API_KEY = "Cohere KEY"


def init_pinecone():
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index_name = "qa-bot"
    
    if index_name in pc.list_indexes().names():
        print(f"Deleting existing index '{index_name}'...")
        pc.delete_index(index_name)
        
    print(f"Creating index '{index_name}' with dimension 4096...")
    pc.create_index(
        name=index_name,
        dimension=4096,  # Cohere embedding dimension
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Change to your preferred region
        )
    )
    return pc, index_name

# Load PDF and extract text
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Generate embeddings using Cohere
def generate_embeddings(documents):
    co = cohere.Client(COHERE_API_KEY)
    embeddings = co.embed(texts=documents).embeddings
    return embeddings

# Store embeddings in Pinecone
def store_embeddings_in_pinecone(embeddings, pc, index_name):
    index = pc.Index(index_name)
    for i, embed in enumerate(embeddings):
        index.upsert([(f'doc_{i}', embed)])
    print("Embeddings stored successfully in Pinecone.")

# Query Pinecone for relevant documents
def query_pinecone(query_embedding, pc, index_name, top_k=5):
    index = pc.Index(index_name)
    query_result = index.query(vector=query_embedding, top_k=top_k)
    return query_result

# Generate answer using Cohere
def generate_answer(query, retrieved_texts):
    co = cohere.Client(COHERE_API_KEY)
    prompt = f"Question: {query}\n\nContext: {' '.join(retrieved_texts)}\n\nAnswer:"
    response = co.generate(prompt=prompt, max_tokens=150)
    return response.generations[0].text.strip()

# Main Streamlit App
def main():
    st.title("Interactive QA Bot")
    
    pc, index_name = init_pinecone()

    # File upload section
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        documents = text.split("\n")
        
        embeddings = generate_embeddings(documents)
        store_embeddings_in_pinecone(embeddings, pc, index_name)
        
        st.success("PDF processed and embeddings stored.")
        
        query = st.text_input("Ask a question about the document:")
        
        if query:
            tokenizer = AutoTokenizer.from_pretrained("cohere/c4")
            model = AutoModel.from_pretrained("cohere/c4")
            tokens = tokenizer(query, return_tensors="pt")
            with torch.no_grad():
                query_embedding = model(**tokens).last_hidden_state.mean(dim=1).squeeze().tolist()
            
            query_result = query_pinecone(query_embedding, pc, index_name)
            retrieved_texts = [documents[int(match.id.split('_')[1])] for match in query_result.matches]
            
            answer = generate_answer(query, retrieved_texts)
            st.write("Answer:", answer)
            st.write("Retrieved Contexts:", retrieved_texts)

if __name__ == "__main__":
    main()
