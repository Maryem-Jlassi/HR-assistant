import ollama
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Chemin vers le fichier CSV
CSV_PATH = "files.csv"  # Assurez-vous d'avoir le bon chemin
MODEL = "minicpm-v"  # Modèle Ollama à utiliser

# Initialisation du modèle d'embedding
embedder = SentenceTransformer('all-MiniLM-L6-v2')
encodings = ["ISO-8859-1", "latin1", "utf-8", "cp1252"]
df = None

# Tentative de chargement avec différents encodages
for enc in encodings:
    try:
        df = pd.read_csv(CSV_PATH, delimiter=",", encoding=enc)
        print(f"Chargé avec succès en utilisant l'encodage : {enc}")
        print(df.head())
        break
    except Exception as e:
        print(f"Échec avec {enc}: {e}")

if df is None:
    print("Impossible de charger le fichier CSV avec les encodages essayés.")
    exit(1)

# Génération des embeddings pour les questions
def generate_embeddings(texts):
    return embedder.encode(texts, convert_to_tensor=False)

# Stockage des embeddings dans FAISS
def store_embeddings_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index

# Recherche de la réponse la plus pertinente
def search_best_match(query, index, questions, responses):
    query_embedding = embedder.encode([query], convert_to_tensor=False)
    D, I = index.search(np.array(query_embedding, dtype='float32'), k=1)
    best_match_index = I[0][0]
    return responses[best_match_index] if best_match_index < len(responses) else "Désolé, je n'ai pas trouvé de réponse."

# Fonction pour générer le prompt pour Ollama
def build_prompt(query, response):
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Vous êtes un assistant RH d'ACTIA. Répondez avec clarté et concision."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
    }

# Fonction principale du chatbot
def main():
    print("Chatbot RH ACTIA - Posez vos questions (tapez 'exit' pour quitter)")
    
    # Chargement des données et création de l'index
    try:
        # Extraction des colonnes par position
        # Basé sur l'affichage: colonne 1 = Question, colonne 3 = Réponse
        print("Nombre total de colonnes dans le CSV:", len(df.columns))
        
        # Utilisation des indices de colonnes plutôt que des noms
        questions = df.iloc[:, 1].tolist()  # Colonne à l'index 1 (2ème colonne)
        responses = df.iloc[:, 3].tolist()  # Colonne à l'index 3 (4ème colonne)
        
        print(f"Nombre de questions chargées: {len(questions)}")
        print(f"Nombre de réponses chargées: {len(responses)}")
        
        # Vérification des premières entrées pour s'assurer que nous avons les bonnes colonnes
        if len(questions) > 0 and len(responses) > 0:
            print("Exemple de question:", questions[0])
            print("Exemple de réponse:", responses[0])
        
        embeddings = generate_embeddings(questions)
        index = store_embeddings_in_faiss(embeddings)
        print("Base de connaissances chargée avec succès !")
    except Exception as e:
        print(f"Erreur de chargement des données : {e}")
        import traceback
        traceback.print_exc()
        return

    # Boucle d'interaction avec l'utilisateur
    while True:
        query = input("\nVotre question : ").strip()
        if query.lower() == "exit":
            print("Au revoir !")
            break
        
        best_response = search_best_match(query, index, questions, responses)
        prompt = build_prompt(query, best_response)
        
        try:
            response = ollama.chat(**prompt)
            print("\nRéponse de l'IA :", response['message']['content'])
        except Exception as e:
            print(f"Erreur avec Ollama : {e}")
            print("Détails:", str(e))

if __name__ == "__main__":
    main()