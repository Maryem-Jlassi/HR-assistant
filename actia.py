import requests
from bs4 import BeautifulSoup
import pandas as pd

# Fonction pour récupérer les offres d'emploi d'ACTIA
def get_actia_jobs():
    url = "https://www.actia.com"  # Vérifiez si l'URL est correcte
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Erreur lors de l'accès à la page")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    job_list = []
    
    for job in soup.find_all("div", class_="job-listing"):  # Adaptez cette classe selon la structure HTML réelle
        title = job.find("h3").text.strip() if job.find("h3") else "N/A"
        location = job.find("span", class_="location").text.strip() if job.find("span", class_="location") else "N/A"
        link = job.find("a")["href"] if job.find("a") else "N/A"
        job_list.append({"Titre": title, "Lieu": location, "Lien": link})
    
    return job_list

# Fonction pour récupérer les avis sur Glassdoor
def get_glassdoor_reviews():
    url = "https://www.glassdoor.fr/Avis/Actia-Avis-E14519.htm"  # Peut nécessiter Selenium si bloqué
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Erreur lors de l'accès à Glassdoor")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    reviews = []
    
    for review in soup.find_all("li", class_="empReview"):  # Adaptez cette classe selon la structure HTML réelle
        rating = review.find("span", class_="rating").text.strip() if review.find("span", class_="rating") else "N/A"
        comment = review.find("p", class_="review-text").text.strip() if review.find("p", class_="review-text") else "N/A"
        reviews.append({"Note": rating, "Commentaire": comment})
    
    return reviews

# Exécuter les fonctions
if __name__ == "__main__":
    jobs = get_actia_jobs()
    reviews = get_glassdoor_reviews()
    
    df_jobs = pd.DataFrame(jobs)
    df_reviews = pd.DataFrame(reviews)
    
    print("Offres d'emploi ACTIA :")
    print(df_jobs.head())
    
    print("\nAvis Glassdoor :")
    print(df_reviews.head())
    
    # Sauvegarde en CSV
    df_jobs.to_csv("actia_jobs.csv", index=False)
    df_reviews.to_csv("actia_reviews.csv", index=False)
    
    print("Données enregistrées en CSV !")
