# Pdf-GPT

This is Langchain project that enables you to interact with any PDFs via Gradio chat interface. It not only fetches relevant answers but also renders relavant page of the PDFs.

# Technologies Used
1. Langchain
2. ChromaDB as vector store
3. OpenAI embeddings
4. OpenAI chat model (gpt-3.5-turbo)
5. Gradio 

# Steps we perform

1. Build a chatbot interface using Gradio
2. Extract texts from pdfs and create embeddings
3. Store embeddings in the Chroma vector database
4. Send query to the backend (Langchain chain)
5. Perform semantic search over texts to find relevant sources of data
6. Send data to LLM (ChatGPT) and receive answers on the chatbot

# How does the end product look like

![Alt Text](https://imgur.com/a/NDHj52m)
