# Code for Extracting data from PDF


import PyPDF2
def extract_text_from_pdf(pdf_path):
        reader=PyPDF2.PdfReader(pdf_path)
        text=''
        for page_num in range(len(reader.pages)):
           page=reader.pages[page_num]
           text=text+page.extract_text()
        return text

# Creating Embeddings
from sentence_transformers import SentenceTransformer

model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def create_embeddings(text_chunks):
    embeddings=model.encode(text_chunks)
    return embeddings
    
#Vector Database
import faiss
def create_faiss_index(embeddings):
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
def search_index_faiss(index,query,k=5):
    distance,indices=index.search(query,k)
    return indices

#query processing
def process_query(query,textchunks,index):
    query_embeddings=model.encode([query])
    indices=search_index_faiss(index,query_embeddings)
    return [textchunks[i] for i in indices[0]]

# Intergating with LLM
import cohere
co=cohere.Client('hvRVpphUaGCoJ63M7PMKMhN04YR9zuNjxqfhnCT5')

def generate_response(prompt):
    response=co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=150
    )
    return response.generations[0].text.strip()

def process_query_with_llm(query,text_chunks,index):
    relevant_chunks=process_query(query,text_chunks,index)
    combined_text="".join(relevant_chunks)
    prompt = f"Based on the following text, answer the question: {query}\n\n{combined_text}"
    return generate_response(prompt)


# Frontend Development
import streamlit as st
st.title("THE RESEARCH PAPERS GUIDE")
uploaded_file=st.file_uploader("Upload any reseach paper",type="PDF")
if uploaded_file is not None:
    text=extract_text_from_pdf(uploaded_file)
    text_chunks=[text[i:i+500] for i in range(0,len(text),500)]
    embeddigs=create_embeddings(text_chunks)
    index=create_faiss_index(embeddigs)
    query=st.text_input("Enter your query")
    if query:
        results=process_query_with_llm(query,text_chunks,index)
        st.write(results)

   
