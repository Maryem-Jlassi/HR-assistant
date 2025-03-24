import ollama
import os
import pandas as pd  # Pour traiter les fichiers CSV
from sentence_transformers import SentenceTransformer
import faiss  # FAISS pour le stockage des vecteurs
import numpy as np

# Constants
CSV_PATH = "files.csv"  # Modifier avec le chemin réel du fichier CSV
MODEL = "minicpm-v"  # Modèle utilisé

# Initialisation du modèle SentenceTransformer pour les embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Fonction pour extraire du texte d'un fichier CSV
def extract_csv_text(csv_path):
    df = pd.read_csv(csv_path,delimiter=";")  # Lecture du fichier CSV
    text_data = []
    
    for _, row in df.iterrows():
        text_data.append(" ".join(map(str, row.values)))  # Concaténer toutes les colonnes en une seule string
    
    return text_data

# Fonction pour générer les embeddings
def generate_query_embedding(query):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    return query_embedding

# Fonction pour générer les embeddings des lignes CSV
def build_embeddings(text_data):
    embeddings = embedder.encode(text_data, convert_to_tensor=True)
    return embeddings

# Stockage des embeddings dans FAISS
def store_embeddings_in_faiss(embeddings):
    embeddings_cpu = embeddings.cpu().detach().numpy()
    dimension = embeddings_cpu.shape[1]
    index = faiss.IndexFlatL2(dimension)
    embeddings_np = np.ascontiguousarray(embeddings_cpu, dtype='float32')
    index.add(embeddings_np)
    return index

# Recherche du contexte pertinent
def search_relevant_context(query, index, text_data):
    query_embedding = generate_query_embedding(query)
    query_embedding_cpu = query_embedding.cpu().detach().numpy().reshape(1, -1)
    D, I = index.search(query_embedding_cpu, k=1)  # Recherche du meilleur match
    relevant_text = text_data[I[0][0]]
    return relevant_text

# Construction de l'index FAISS
def build_and_store_index(csv_path):
    text_data = extract_csv_text(csv_path)
    embeddings = build_embeddings(text_data)
    index = store_embeddings_in_faiss(embeddings)
    return index, text_data
def build_prompt_with_context(query: str, index, chunks, image_path: str = None) -> dict:
    """
    Build a RAG prompt by combining the query with the most relevant context from the PDF.
    """
    # Retrieve the most relevant context for the query
    relevant_context = search_relevant_context(query, index, chunks)
    
    messages = [
        {
            "role": "user",
            "content": (
                "You are an hr expert As an HR expert, you are responsible for evaluating candidate CVs and determining their suitability for a specific job opening."
                "Your task is to design an efficient scoring system that assesses CVs based on their alignment with the provided job description and job post. "
                "Do not use any external knowledge or make assumptions beyond the provided content. "
                "You will receive a query along with context extracted from a PDF, and optionally, an image for reference. "
                "Your answer should strictly reference and use the provided context. "
                "The structure of your response should be:\n\n"
                "1. **Domain Detection**: Analyze job description to identify domain from:"
                 "- Acquisition | Financial Services | Administrative Services "
                 "- Transport/Logistics | IT/Business Intelligence | HR/Recruitment "
                 "- Controlling/Reporting | Other Opportunities"

                "2. **Weight Assignment**: Use these domain weights:"
                 "{weights}"

                "3. **CV Analysis**: Calculate similarity score using:"
                "Score = Σ(category_similarity * category_weight)"

                "4. **Output**: JSON with:"
                "- Domain detected"
                "- Final percentage score"
                "- Weights used"
                "- Key matches/mismatches\n\n"

                 "Be concise and domain-focused."""

                "PROMPT = **Job Description**:  {job_description}"

                "**CV**:  {cv_text}"

                " Analyze match using correct domain weights."""
                "5. If an image is provided, integrate its content only if necessary to support the answer.\n\n"
                "Keep the response concise and focused on the key points from the context."
            )
        },
        {
            "role": "system",
            "content": f"Relevant context from the PDF: {relevant_context}"
        }
    ]

    
    messages.append({
        "role": "user",
        "content": query
    })

    return {
        "model": MODEL,
        "messages": messages
    }

# Main function
def main():
    print("Welcome to the AI Chat Assistant with PDF and optional image support!")
    print(f"Using fixed PDF file: {CSV_PATH}")
    print("You can optionally provide an image for additional context.")
    print("Type 'exit' to quit the chat.")

    if not os.path.exists(CSV_PATH):
        print(f"Error: The mandatory PDF file '{CSV_PATH}' was not found. Please check the file path.")
        return

    # Build and store the index for the PDF
    index, chunks = build_and_store_index(CSV_PATH)
    print("Index built and ready for searching relevant context!")
    
    print("\nInteractive Chat Started! Ask your questions below.")
    
    while True:
        query = input("\nYour Query: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        
        prompt = build_prompt_with_context(query, index, chunks)
        try:
            response = ollama.chat(**prompt)
            print("\nAI Response:")
            print(response['message']['content'])
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")

if __name__ == "__main__":
    main()
