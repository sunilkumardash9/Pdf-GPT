# Pdf-GPT

This is a Langchain project that enables you to interact with any PDFs via the Gradio chat interface. It not only fetches relevant answers but also renders relevant pages of the PDFs.

# Technologies Used
1. Langchain
2. ChromaDB as vector store
3. OpenAI embeddings
4. OpenAI chat model (gpt-3.5-turbo)
5. Gradio 

# Steps performed

1. Build a chatbot interface using Gradio
2. Extract texts from pdfs and create embeddings
3. Store embeddings in the Chroma vector database
4. Send query to the backend (Langchain chain)
5. Perform a semantic search over texts to find relevant sources of data
6. Send data to LLM (ChatGPT) and receive answers on the chatbot

# Code walkthrough guide

https://www.analyticsvidhya.com/blog/2023/05/build-a-chatgpt-for-pdfs-with-langchain/


# How does the end product look like

![alt text](https://github.com/sunilkumardash9/Pdf-GPT/blob/main/Resources/Screenshot%20from%202023-05-10%2022-07-20.png?raw=true)

# Demo Video


https://github.com/sunilkumardash9/Pdf-GPT/assets/47926185/5d5f9f43-fff9-4a3c-a5af-5f0b9c7904fd


