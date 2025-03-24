import ollama
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader
import re
import torch

# Chemins des fichiers
CSV_PATH = "files.csv"
DOCS_DIR = "actia_docs/"  # Dossier contenant les documents ACTIA (PDF, TXT, etc.)
MODEL = "minicpm-v"  # Modèle Ollama à utiliser

# Initialisation du modèle d'embedding
#embedder = SentenceTransformer('all-MiniLM-L6-v2')
embedder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Fonction pour extraire le texte des PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Fonction pour extraire le texte des fichiers TXT
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

# Fonction pour découper le texte en chunks (morceaux) pour un meilleur traitement
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Fonction pour charger tous les documents et créer des chunks
def load_all_documents():
    all_chunks = []
    
    # Charger les données du CSV
    try:
        df = pd.read_csv(CSV_PATH, delimiter=";", encoding="ISO-8859-1")
        # Combinaison des questions et réponses
        for i, row in df.iterrows():
            question = row.iloc[1]  # Colonne Question (index 1)
            answer = row.iloc[3]    # Colonne Réponse (index 3)
            all_chunks.append(f"Question: {question} Réponse: {answer}")
    except Exception as e:
        print(f"Erreur lors du chargement du CSV: {e}")
    
    # Charger les documents PDF et TXT
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            filepath = os.path.join(DOCS_DIR, filename)
            try:
                if filename.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(filepath)
                    text_chunks = chunk_text(text)
                    all_chunks.extend(text_chunks)
                    print(f"Document PDF chargé: {filename}, {len(text_chunks)} chunks extraits")
                
                elif filename.lower().endswith('.txt'):
                    text = extract_text_from_txt(filepath)
                    text_chunks = chunk_text(text)
                    all_chunks.extend(text_chunks)
                    print(f"Document TXT chargé: {filename}, {len(text_chunks)} chunks extraits")
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {filename}: {e}")
    else:
        print(f"Le répertoire {DOCS_DIR} n'existe pas. Veuillez le créer et y ajouter vos documents.")
    
    return all_chunks

# Génération des embeddings et création de l'index FAISS
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index, chunks

# Fonction pour rechercher les chunks les plus pertinents
def retrieve_relevant_chunks(query, index, chunks, k=10):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), k=k)
    
    # Récupérer les chunks pertinents
    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(chunks):
            relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

# Construction du prompt pour Ollama avec RAG
def build_rag_prompt(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": """Vous êtes un assistant RH spécialisé pour ACTIA. 
Répondez avec clarté et concision, en vous basant uniquement sur les informations fournies 
dans le contexte. Si vous ne trouvez pas l'information dans le contexte ou Si l'information est insuffisante, reformulez en demandant plus de précisions, 
mais essayez d'exploiter ce qui est disponible. Évitez d'inventer des informations. 
Soyez professionnel et informatif."""},
            {"role": "user", "content": f"Contexte: {context}\n\nQuestion: {query}"}
        ]
    }

# Fonction principale du chatbot
def main():
    print("Chargement de la base de connaissances ACTIA...")
    
    # Chargement de tous les documents
    all_chunks = load_all_documents()
    
    if not all_chunks:
        print("Aucune donnée trouvée. Veuillez vérifier vos fichiers source.")
        return
    
    print(f"Base de connaissances chargée avec succès! {len(all_chunks)} segments de texte disponibles.")
    
    # Création de l'index FAISS
    index, chunks = create_faiss_index(all_chunks)
    
    print("Chatbot RH ACTIA - Posez vos questions (tapez 'exit' pour quitter)")
    
    # Boucle d'interaction
    while True:
        query = input("\nVotre question : ").strip()
        if query.lower() == "exit":
            print("Au revoir !")
            break
        
        # Récupération des chunks pertinents
        relevant_chunks = retrieve_relevant_chunks(query, index, chunks)
        
        # Construction du prompt RAG
        prompt = build_rag_prompt(query, relevant_chunks)
        
        try:
            response = ollama.chat(**prompt)
            print("\nRéponse de l'IA :", response['message']['content'])
        except Exception as e:
            print(f"Erreur avec Ollama : {e}")

if __name__ == "__main__":
    main()