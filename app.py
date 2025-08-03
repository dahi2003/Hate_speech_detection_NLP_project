
from flask import Flask, request, render_template
import pickle
import re
import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def transcribe_video(file_path):
    audio_path = os.path.join(app.config["UPLOAD_FOLDER"], "temp_audio.wav")
    clip = VideoFileClip(file_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    return transcribe_audio(audio_path)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        text_input = request.form.get("tweet", "").strip()
        audio_file = request.files.get("audio")
        video_file = request.files.get("video")

        transcript = text_input

        if audio_file and audio_file.filename != "":
            audio_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(audio_file.filename))
            audio_file.save(audio_path)
            transcript = transcribe_audio(audio_path)

        elif video_file and video_file.filename != "":
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(video_file.filename))
            video_file.save(video_path)
            transcript = transcribe_video(video_path)

        cleaned = clean_tweet(transcript)
        if cleaned:
            vectorized = tfidf.transform([cleaned])
            label = model.predict(vectorized)[0]
            labels = {0: "Hate Speech", 1: "Offensive Language", 2: "Neither"}
            prediction = labels[label]
        else:
            prediction = "Could not extract meaningful text."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)