import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
import os

import fitz
from PIL import Image

COUNT,N = 0,0
chat_history = []
chain = ''
enable_box = gr.Textbox.update(value=None,placeholder= 'Upload your OpenAI API key',interactive=True)
disable_box = gr.Textbox.update(value = 'OpenAI API key is Set',interactive=False)
#os.environ['OPENAI_API_KEY'] = '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'

def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    return disable_box
def enable_api_box():
    return enable_box

def add_text(history, text):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')] 
    return history

def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        raise gr.Error('Upload your OpenAI API key')

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    embeddings = OpenAIEmbeddings()
    
    pdfsearch = Chroma.from_documents(documents, embeddings, metadatas=[{"source": f"{i}-pl"} for i in range(len(documents))])

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3), 
                                                  retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                                  return_source_documents=True,)
    return chain

def generate_response(history, query, btn):
    global COUNT, N, chat_history, chain
    
    if not btn:
        raise gr.Error(message='Upload a PDF')
    if COUNT == 0:
            chain = process_file(btn)
            COUNT += 1
    
    result = chain({"question": query, 'chat_history':chat_history},return_only_outputs=True)
    chat_history += [(query, result["answer"])]
    N = list(result['source_documents'][0])[1][1]['page']

    for char in result['answer']:
       history[-1][-1] += char
       yield history,''


def render_first(btn):
    
    doc = fitz.open(btn.name)
    page = doc[0]

    #Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

def render_file(file):
    global N
    doc = fitz.open(file.name)
    page = doc[N]
    #Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image


with gr.Blocks() as demo:

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=0.8):
                api_key = gr.Textbox(placeholder='Enter OpenAI API key', show_label=False, interactive=True).style(container=False)
            with gr.Column(scale=0.2):
                change_api_key = gr.Button('Change Key')
        with gr.Row():           
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select' ).style(height=680)
    with gr.Row():
        with gr.Column(scale=0.70):
            txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                    ).style(container=False)
        with gr.Column(scale=0.15):
            submit_btn = gr.Button('submit')
        with gr.Column(scale=0.15):
            btn = gr.UploadButton("📁 upload a PDF", file_types=[".pdf"]).style()
    
    api_key.submit(fn=set_apikey, inputs=[api_key], outputs=[api_key])
    change_api_key.click(fn= enable_api_box,outputs=[api_key])
    btn.upload(fn=render_file, inputs=[btn], outputs=[show_img],)
    
    submit_btn.click(fn=add_text, inputs=[chatbot,txt], outputs=[chatbot, ], queue=False).success(fn=generate_response,inputs = [chatbot, txt, btn],
                                    outputs = [chatbot,txt]).success(fn=render_file,inputs = [btn], outputs=[show_img])

    txt.submit(fn=add_text, inputs=[chatbot,txt], outputs=[chatbot, ], queue=False).success(fn=generate_response,inputs = [chatbot, txt, btn],
                                    outputs = [chatbot,txt]).success(fn=render_file,inputs = [btn], outputs=[show_img])


demo.queue()
if __name__ == "__main__":
    demo.launch()  
