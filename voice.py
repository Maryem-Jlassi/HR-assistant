import ollama
import os
import pandas as pd
import faiss
import numpy as np
import torch
import speech_recognition as sr
import pyttsx3
import tkinter as tk
from tkinter import scrolledtext
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
# Chemins des fichiers et modèles
CSV_PATH = "files.csv"
DOCS_DIR = "actia_docs/"
MODEL = "minicpm-v"

# Initialisation de l'embedding
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Initialisation de la synthèse vocale
engine = pyttsx3.init()

# Initialisation de l'interface graphique
window = tk.Tk()
window.title("Chatbot RH ACTIA - Assistant Vocal")
window.geometry("500x600")

status_label = tk.Label(window, text="Prêt", font=("Arial", 12))
status_label.pack(pady=10)

text_area = scrolledtext.ScrolledText(window, width=50, height=20, font=("Arial", 12))
text_area.pack(padx=10, pady=5)

text_entry = tk.Entry(window, font=("Arial", 12), width=40)
text_entry.pack(pady=5)
# Extraction du texte depuis les fichiers PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Extraction du texte depuis les fichiers TXT
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# Découpe du texte en morceaux pour RAG
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

# Chargement de tous les documents et stockage en mémoire
def load_all_documents():
    all_chunks = []
    
    # Chargement des données CSV
    try:
        df = pd.read_csv(CSV_PATH, delimiter=";", encoding="ISO-8859-1")
        for _, row in df.iterrows():
            question, answer = row.iloc[1], row.iloc[3]
            all_chunks.append(f"Question: {question} Réponse: {answer}")
    except Exception as e:
        print(f"Erreur chargement CSV: {e}")

    # Chargement des documents (PDF/TXT)
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            filepath = os.path.join(DOCS_DIR, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    all_chunks.extend(chunk_text(extract_text_from_pdf(filepath)))
                elif filename.lower().endswith('.txt'):
                    all_chunks.extend(chunk_text(extract_text_from_txt(filepath)))
            except Exception as e:
                print(f"Erreur chargement {filename}: {e}")
    return all_chunks
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype='float32'))
    return index, chunks
def retrieve_relevant_chunks(query, index, chunks, k=5):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=k)
    return [chunks[idx] for idx in indices[0] if idx < len(chunks)]
def build_rag_prompt(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant RH spécialisé ACTIA..."},
            {"role": "user", "content": f"Contexte: {context}\n\nQuestion: {query}"}
        ]
    }

def get_chatbot_response(query):
    relevant_chunks = retrieve_relevant_chunks(query, faiss_index, chunks_db)
    prompt = build_rag_prompt(query, relevant_chunks)
    
    try:
        response = ollama.chat(**prompt)
        return response['message']['content']
    except Exception as e:
        return f"Erreur avec Ollama: {e}"
def speak(text):
    """Convertir le texte en voix et afficher dans l'interface"""
    engine.say(text)
    engine.runAndWait()
    text_area.insert(tk.END, f"Assistant: {text}\n")
    text_area.yview(tk.END)

def listen():
    """Capturer la voix et traiter la question"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        status_label.config(text="Écoute...", fg="blue")
        try:
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio, language="fr-FR")
            text_area.insert(tk.END, f"Vous (Voix): {command}\n")
            process_chat(command)
        except sr.UnknownValueError:
            text_area.insert(tk.END, "Assistant: Je n'ai pas compris.\n")
        except sr.RequestError:
            text_area.insert(tk.END, "Assistant: Erreur réseau.\n")
        status_label.config(text="Prêt", fg="black")
def process_chat(query):
    response = get_chatbot_response(query)
    text_area.insert(tk.END, f"Assistant: {response}\n")
    text_area.yview(tk.END)
    speak(response)

def start_listening():
    threading.Thread(target=listen).start()

def manual_input():
    command = text_entry.get()
    if command:
        text_area.insert(tk.END, f"Vous (Texte): {command}\n")
        text_entry.delete(0, tk.END)
        process_chat(command)

# Boutons d’interaction
listen_button = tk.Button(window, text="🎤 Écouter", font=("Arial", 12), command=start_listening)
listen_button.pack(pady=5)

submit_button = tk.Button(window, text="📩 Envoyer", font=("Arial", 12), command=manual_input)
submit_button.pack(pady=5)

exit_button = tk.Button(window, text="❌ Quitter", font=("Arial", 12), command=window.quit)
exit_button.pack(pady=10)

# Charger la base de données
chunks_db = load_all_documents()
faiss_index, _ = create_faiss_index(chunks_db)

window.mainloop()
