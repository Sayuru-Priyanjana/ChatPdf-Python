from flask import Flask, render_template, request, redirect, url_for, flash
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import csv
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Variable to store filenames between requests
uploaded_filenames = []

# Function to extract text from PDF files
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, \nend of the answer relevant page
     number and pdf name. If asked to summarize, do it. If the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input and store chat history
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    answer = response["output_text"]
    save_chat_history(user_question, answer, uploaded_filenames)
    
    return answer

# Function to save chat history to CSV
def save_chat_history(question, answer, pdf_files):
    with open("chat_history.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, answer, ", ".join(pdf_files)])

# Function to view chat history
def view_chat_history():
    history = []
    if os.path.exists("chat_history.csv"):
        with open("chat_history.csv", mode="r") as file:
            reader = csv.reader(file)
            for idx, row in enumerate(reader):
                if len(row) == 4:
                    history.append({"id": idx, "datetime": row[0], "question": row[1], "answer": row[2], "pdfs": row[3]})
    return history

# Function to delete a chat entry from history
def delete_chat_history(entry_id):
    if os.path.exists("chat_history.csv"):
        with open("chat_history.csv", mode="r") as file:
            lines = file.readlines()
        with open("chat_history.csv", mode="w") as file:
            for idx, line in enumerate(lines):
                if idx != entry_id:
                    file.write(line)

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_filenames
    if request.method == "POST":
        pdf_files = request.files.getlist("pdf_files")
        uploaded_filenames = []
        for pdf_file in pdf_files:
            if pdf_file and pdf_file.filename.endswith(".pdf"):
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(pdf_path)
                raw_text = get_pdf_text(pdf_path)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                uploaded_filenames.append(pdf_file.filename)
        flash("PDF files processed successfully.", "success")
    return render_template("index.html", filenames=uploaded_filenames)

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question")
    if user_question:
        answer = user_input(user_question)
        return render_template("index.html", question=user_question, answer=answer, filenames=uploaded_filenames)
    return redirect(url_for("index"))

@app.route("/history", methods=["GET", "POST"])
def history():
    if request.method == "POST":
        entry_id = int(request.form.get("entry_id"))
        delete_chat_history(entry_id)
        flash("Chat history entry deleted.", "success")
    chat_history = view_chat_history()
    return render_template("history.html", chat_history=chat_history)

if __name__ == "__main__":
    app.run(debug=True)
