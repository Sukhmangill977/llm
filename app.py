from typing import List
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import uvicorn
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
import requests

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say "answer is not available in the context"; don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def download_file(url, local_filename=None):
    # If no filename is provided, use the name from the URL
    if not local_filename:
        local_filename = url.split('/')[-1].split('?')[0]
    
    try:
        # Send HTTP GET request to the URL
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check if the request was successful
            # Open a local file in binary write mode
            with open(local_filename, 'wb') as file:
                # Stream the content and write it to the file in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive new chunks
                        file.write(chunk)
        print(f"File downloaded successfully: {local_filename}")
    except Exception as e:
        print(f"Error downloading file: {e}")

    return local_filename

@app.post("/ask")
async def ask_question(pdf_urls: str = Form(...), question: str = Form(...)):
    pdf_urls_list = pdf_urls.split(',')
    print("pdf_urls:", pdf_urls)
    print("Processing PDF files...")
    answers = []
    
    for i, url in enumerate(pdf_urls_list):
        pdf_name = f"From pdf{i+1}"  # Indicate the source PDF
        pdf_file = download_file(url)
        raw_text = get_pdf_text(pdf_file)
        os.remove(pdf_file)
        
        # Process the extracted text
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Load vector store and find the most relevant chunks
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question, k=10)  # Increase 'k' to get more relevant chunks

        # Get the answer using the conversational chain
        chain = get_conversational_chain()
        for doc in docs:
            response = chain({"input_documents": [doc], "question": question}, return_only_outputs=True)
            output_text = response['output_text'].strip()

            # Check if the output is meaningful and not just "Answer is not available in the context"
            if "Answer is not available in the context" not in output_text:
                answers.append(f"{pdf_name}: {output_text}")

    # Combine all answers into one response
    final_answer = "\n\n".join(answers)

    if not final_answer:
        final_answer = "Answer is not available in the provided documents."

    # Return the answer as JSON
    return JSONResponse(content={"answer": final_answer})

    # Return the answer as JSON
    return JSONResponse(content={"answer": final_answer})

# Run the FastAPI app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get port from environment, default to 8000
    uvicorn.run(app, host="127.0.0.1", port=port)

