This Project aims in developing a RAG Application using Streamlit as the Frontend and this application aims in interpreting the research papers.
The application asks the user to upload a pdf file ,, which can be research paper and provides provision to enter a query.
Based on the query it will display the result.The application used PyPDF2.PDFReader to extract data from the PDF ,sentence-transformer to create embeddings, faiss CPU to store the vector database
and cohere's command-r-plus model to generate a detailed response for the query & display the same.
