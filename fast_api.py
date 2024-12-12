from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
#from fastapi.staticfiles import StaticFiles
from fastapi.requests import Request
import joblib
import scipy.sparse as sp

# Charger les objets sauvegardés
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")
nb_model = joblib.load("spam_model.pkl")

# Initialiser FastAPI
app = FastAPI()

# Configurer les templates Jinja2
templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Route principale
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint pour prédire
@app.post("/predict/", response_class=HTMLResponse)
def predict(request: Request, email: str = Form(...)):
    try:
        # Préparation des données
        text_vectorized = tfidf_vectorizer.transform([email])
        length = len(email)
        num_special_chars = sum(1 for char in email if not char.isalnum())
        numerical_data = scaler.transform([[length, num_special_chars]])
        features = sp.hstack((text_vectorized, numerical_data))

        # Prédiction
        prediction = nb_model.predict(features)
        result = "SPAM" if prediction[0] == 1 else "HAM"
        proba = nb_model.predict_proba(features)

        # Retourner la page avec les résultats
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,
            "email": email,
            "probability_spam": round(proba[0][1] * 100, 2),
            "probability_ham": round(proba[0][0] * 100, 2)
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": str(e),
            "email": email
        })
