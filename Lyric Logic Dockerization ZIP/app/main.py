from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import keras.models as models
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from keras.utils import (
    pad_sequences,
)
import pickle

import os
import psycopg2


# Initialize the NLTK library
nltk.download("stopwords")
nltk.download("wordnet")


# List of artists
artists = [
    "Eminem a.k.a. Slim Shady a.k.a. Marshall Mathers",
    "Taylor Swift a.k.a. Kanye's nemesis",
    "Drake a.k.a. Kendrick's number one fan",
    "Beyonce a.k.a. Jay-Z's wife",
    "Rihanna a.k.a. Eminem's love interest",
    "Lady Gaga (Who still listens to her?)",
    "Justin Bieber a.k.a. the baby",
    "Coldplay a.k.a. the one with the yellow stars",
    "Katy Perry a.k.a. the one with the fireworks",
    "Nicki Minaj a.k.a. tbh idrc",
    "Ariana Grande a.k.a. the one with the ponytail",
    "Ed Sheeran a.k.a. the ginger",
    "Dua Lipa a.k.a. elfnana",
]

artist_images = [
    "/static/images/em.jpg",
    "/static/images/swift.jpg",
    "/static/images/drake.jpg",
    "/static/images/beyonce.jpg",
    "/static/images/rihanna.jpg",
    "/static/images/lg.jpg",
    "/static/images/jb.jpg",
    "/static/images/chris.jpg",
    "/static/images/kp.jpg",
    "/static/images/nicki.jpg",
    "/static/images/ariana.jpg",
    "/static/images/edsheeran.jpg",
    "/static/images/dualipa.jpg",
]

# Set up the necessary tools for text preprocessing
punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
max_sequence_length = 300

# Get the tokenizer
with open("./models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Initialize the FastAPI app
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the templates
templates = Jinja2Templates(directory="templates")

# Initialize global variables
previous_model = ""
previous_lyrics = ""
previous_prediction = None


def preprocess_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = "".join([l for l in lyrics if l not in punctuation])
    lyrics = lyrics.replace(r"[^a-zA-Z0-9 ]", "")
    lyrics = " ".join([word for word in lyrics.split() if word not in stop_words])
    lyrics = " ".join(lemmatizer.lemmatize(word) for word in lyrics.split())
    lyrics = " ".join(stemmer.stem(word) for word in lyrics.split())

    lyrics = [lyrics]
    lyrics = tokenizer.texts_to_sequences(lyrics)
    lyrics = pad_sequences(lyrics, maxlen=max_sequence_length)

    return lyrics


# Global variable for the database connection
connection = None
cursor = None

# Database connection settings from environment variables
db_config = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),  # 'db' is the hostname from docker-compose
    "port": "5432",
}


@app.on_event("startup")
def startup_event():
    global connection, cursor
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()

        # Create the table if it doesn't exist
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255),
                lyrics TEXT,
                artist VARCHAR(255)
            )
            """
        )

        print("Connected to the database and cursor initialized successfully!")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        connection = None
        cursor = None


@app.on_event("shutdown")
def shutdown_event():
    global connection, cursor
    if cursor:
        cursor.close()
        print("Cursor closed.")
    if connection:
        connection.close()
        print("Database connection closed.")


@app.get("/db-status")
def db_status():
    if connection and cursor:
        return {"status": "Connected"}
    else:
        return {"status": "Not connected"}


@app.get("/", response_class=HTMLResponse)
def predict(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": "Please enter some lyrics!",
            "artist_image": "/static/images/cover.jpg",
        },
    )


@app.post("/", response_class=HTMLResponse)
def predict(request: Request, model_name: str = Form(...), lyrics: str = Form(...)):

    global previous_model
    global previous_lyrics
    global previous_prediction

    lyrics = lyrics.strip()

    if len(lyrics) == 0:

        cursor.execute(
            """
            INSERT INTO detections (model_name, lyrics, artist)
            VALUES (%s, %s, %s)
            """,
            (
                model_name,
                "No lyrics",
                "No Artist",
            ),
        )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": "Please enter some lyrics!",
                "artist_image": "/static/images/cover.jpg",
            },
        )

    if previous_model == model_name and previous_lyrics == lyrics:
        if previous_prediction is not None:

            cursor.execute(
                """
                INSERT INTO detections (model_name, lyrics, artist)
                VALUES (%s, %s, %s)
                """,
                (
                    model_name,
                    lyrics,
                    artists[np.argmax(previous_prediction)],
                ),
            )

            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": artists[np.argmax(previous_prediction)],
                    "artist_image": artist_images[np.argmax(previous_prediction)],
                },
            )

    if model_name == "cnn_glove":
        model = models.load_model(
            "./models/cnn_glove.h5",
            compile=False,
        )
    elif model_name == "cnn_learned":
        model = models.load_model(
            "./models/cnn_learnable.h5",
            compile=False,
        )
    elif model_name == "lstm_learned":
        model = models.load_model(
            "./models/lstm_learnable.h5",
            compile=False,
        )
    else:
        model = models.load_model(
            "./models/lstm_learnable.h5",
            compile=False,
        )

    previous_model = model_name
    previous_lyrics = lyrics

    preprocessed_lyrics = preprocess_lyrics(lyrics)

    prediction = model.predict(preprocessed_lyrics)
    predicted_artist = artists[np.argmax(prediction)]

    previous_prediction = prediction

    cursor.execute(
        """
        INSERT INTO detections (model_name, lyrics, artist)
        VALUES (%s, %s, %s)
        """,
        (
            model_name,
            lyrics,
            predicted_artist,
        ),
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": predicted_artist,
            "artist_image": artist_images[np.argmax(prediction)],
        },
    )
