from typing import Any
import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
from PIL import Image
import chromadb
import re
import uuid
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la clave de API de OpenAI desde las variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

enable_box = gr.Textbox.update(value=None, placeholder='Upload your OpenAI API key', interactive=True)
disable_box = gr.Textbox.update(value='OpenAI API key is Set', interactive=False)

def set_apikey(api_key: str):
    app.OPENAI_API_KEY = api_key
    return disable_box

def enable_api_box():
    return enable_box

def add_text(history, text: str):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]
    return history

class my_app:
    def __init__(self, OPENAI_API_KEY: str = None) -> None:
        self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0

    def __call__(self, file: str) -> Any:
        # Rebuild the chain each time a new file is processed
        self.chain = self.build_chain(file)
        return self.chain

    def chroma_client(self):
        # Create a Chroma client
        client = chromadb.Client()
        # Create a collection
        collection = client.get_or_create_collection(name="my-collection")
        return client

    def process_file(self, file: str):
        # Define the new file name
        new_file_name = "libro.pdf"
        
        # Get the current file name and its directory
        directory = os.path.dirname(file.name)
        current_file_name = os.path.basename(file.name)
        
        # Define the new file path
        new_file_path = os.path.join(directory, new_file_name)
        
        # Rename the file
        os.rename(file.name, new_file_path)
        
        # Load the renamed file
        loader = PyPDFLoader(new_file_path)
        documents = loader.load()
        
        return documents, new_file_name
    
    def build_chain(self, file: str):
        documents, file_name = self.process_file(file)
        # Load embeddings model
        embeddings = OpenAIEmbeddings(openai_api_key=self.OPENAI_API_KEY)
        pdfsearch = Chroma.from_documents(documents, embeddings, collection_name=file_name)
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.0, openai_api_key=self.OPENAI_API_KEY),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )
        return chain

def get_response(history, query, file):
    if not file:
        raise gr.Error(message='Upload a PDF')
    chain = app(file)
    result = chain({"question": query, 'chat_history': app.chat_history}, return_only_outputs=True)
    app.chat_history += [(query, result["answer"])]
    app.N = list(result['source_documents'][0])[1][1]['page']
    for char in result['answer']:
        history[-1][-1] += char
        yield history, ''

def render_file(file):
    doc = fitz.open(file.name)
    page = doc[app.N]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

def render_first(file):
    doc = fitz.open(file.name)
    page = doc[0]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image, []

app = my_app()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(placeholder='Enter OpenAI API key', show_label=False, interactive=True).style(container=False)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')
        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select').style(height=680)
    with gr.Row():
        with gr.Column(scale=0.60):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
        with gr.Column(scale=0.20):
            submit_btn = gr.Button('Submit')
        with gr.Column(scale=0.20):
            btn = gr.UploadButton("📁 Upload a PDF", file_types=[".pdf"]).style()
    
    api_key.submit(
        fn=set_apikey,
        inputs=[api_key],
        outputs=[api_key]
    )
    change_api_key.click(
        fn=enable_api_box,
        outputs=[api_key]
    )
    btn.upload(
        fn=render_first,
        inputs=[btn],
        outputs=[show_img, chatbot]
    )
    
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot]
    ).success(
        fn=get_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
