import streamlit as st
import spacy
import pandas as pd
import pyttsx3
import speech_recognition as sr
from googletrans import Translator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from pymongo import MongoClient
import bcrypt

# MongoDB Setup
MONGO_URI = "mongodb://localhost:27017/"
client = MongoClient(MONGO_URI)
db = client["careconnect"]
users_collection = db["users"]
history_collection = db["chat_history"]

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load CSV dataset
symptoms_df = pd.read_csv(r"C:\\Users\\Sanjana\\OneDrive\\Desktop\\CareConnect\\symptoms_with_doctors_hospitals.csv")

# KNN Model Setup
def prepare_data():
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(symptoms_df['symptom'].apply(lambda x: x.lower()))
    y = symptoms_df['disease']
    return X, y, vectorizer

X, y, vectorizer = prepare_data()
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)

# Classify Disease
def classify_disease(user_symptoms):
    processed_symptoms = ' '.join(
        [token.text for token in nlp(user_symptoms.lower()) if not token.is_stop]
    )
    user_symptoms_processed = vectorizer.transform([processed_symptoms])
    predicted_disease = knn_model.predict(user_symptoms_processed)[0]
    disease_info = symptoms_df[symptoms_df['disease'] == predicted_disease].iloc[0]
    return {
        'disease': disease_info['disease'],
        'doctor': disease_info['doctor'],
        'hospital': disease_info['hospital'],
        'severity': disease_info['severity']
    }

# Hash Passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(hashed_password, password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

# User Authentication Functions
def sign_up():
    st.sidebar.title("Sign Up")
    username = st.sidebar.text_input("Username", key="signup_username")
    password = st.sidebar.text_input("Password", type="password", key="signup_password")
    if st.sidebar.button("Sign Up"):
        if users_collection.find_one({"username": username}):
            st.sidebar.error("Username already exists!")
        else:
            hashed_pw = hash_password(password)
            users_collection.insert_one({"username": username, "password": hashed_pw})
            st.sidebar.success("Sign-up successful! Please log in.")

def login_user():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")
    if st.sidebar.button("Login"):
        user = users_collection.find_one({"username": username})
        if user and check_password(user["password"], password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success("Login successful!")
        else:
            st.sidebar.error("Invalid username or password!")

# Chat History
def save_chat_history(username, input_text, response):
    history_collection.insert_one({
        "username": username,
        "input": input_text,
        "response": response
    })

def get_chat_history(username):
    return list(history_collection.find({"username": username}))

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'disease_info' not in st.session_state:
    st.session_state.disease_info = None

# Voice Assistant Setup
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening for voice input...")
        audio = recognizer.listen(source)
        try:
            voice_input = recognizer.recognize_google(audio)
            st.write(f"You said: {voice_input}")
            return voice_input
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            st.write("Sorry, the service is unavailable.")
            return ""

# Language Selection
languages = {
    'English': 'en', 'Spanish': 'es', 'French': 'fr', 
    'German': 'de', 'Italian': 'it'
}

selected_language = st.sidebar.selectbox("Choose Language", list(languages.keys()))
st.sidebar.write(f"Current Language: {selected_language}")

# Translate Text
def translate_text(text, lang='en'):
    try:
        if text and isinstance(text, str):
            translator = Translator()
            translated = translator.translate(text, src='en', dest=languages.get(lang, 'en')).text
            return translated
        else:
            return ""
    except Exception as e:
        st.write(f"Error in translation: {e}")
        return text

# Display Login/Sign-Up
if not st.session_state.logged_in:
    st.sidebar.header("Authentication")
    mode = st.sidebar.radio("Choose Action", ["Login", "Sign Up"])
    if mode == "Sign Up":
        sign_up()
    else:
        login_user()
else:
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose Page", ["Home", "Chat History"])

    if page == "Home":
        st.markdown("""
        <div style='background-color: #4CAF50; padding: 10px; border-radius: 5px; text-align: center; width: 80%; margin: auto;'>
            <h1 style='color: white; font-size: 24px;'>MedOracle:"Where Medical Insights Meet Precision"</h1>
        </div>
        """, unsafe_allow_html=True)

        st.image("C:/Users/Sanjana/OneDrive/Desktop/CareConnect/logo1.jpg", width=300)

        st.sidebar.markdown("""
        <div style="background-color: #4CAF50; padding: 8px; border-radius: 3px; border: 1px solid #4CAF50;">
            <h2 style="color: white; text-align: center; font-size: 18px;">Welcome, {}</h2>
        </div>
        <br>
        """.format(st.session_state.username), unsafe_allow_html=True)

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False

        st.markdown("""
        <div style='text-align: center; border: 2px solid #4CAF50; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h3 style='font-weight: bold;'>Enter your symptoms below:</h3>
            <p style='font-style: italic;'>Provide as much detail as possible to get accurate recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.text_area("Describe your symptoms:", height=150)
        if st.button("Use Voice Input"):
            user_input = listen()

        if user_input:
            translated_input = translate_text(user_input, selected_language)

            if translated_input:
                disease_info = classify_disease(translated_input)
                disease_info_translated = {key: translate_text(value, selected_language) for key, value in disease_info.items()}

                save_chat_history(st.session_state.username, user_input, disease_info_translated)
                st.session_state.disease_info = disease_info_translated

                st.markdown(f"""
                <div style='padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; margin-top: 20px;'>
                    <h2 style='color: black;'>Results:</h2>
                    <div style='padding: 10px; border-radius: 5px; margin-bottom: 10px;'>
                        <strong>Disease:</strong> {disease_info_translated['disease']}<br>
                        <strong>Severity:</strong> {disease_info_translated['severity']}<br>
                        <strong>Recommended Doctor:</strong> {disease_info_translated['doctor']}<br>
                        <strong>Recommended Hospital:</strong> {disease_info_translated['hospital']}<br>
                    </div>
                    <h4 style='color: black;'>Stay safe and follow the doctor’s advice!</h4>
                </div>
                """, unsafe_allow_html=True)

                if st.session_state.disease_info:
                    result_text = f"Disease: {disease_info_translated['disease']}. Severity: {disease_info_translated['severity']}. Doctor: {disease_info_translated['doctor']}. Hospital: {disease_info_translated['hospital']}."
                    speak(result_text)
    elif page == "Chat History":
        st.markdown("""
        <div style='background-color: #f1f1f1; padding: 20px; border-radius: 10px;'>
            <h2>Chat History</h2>
        </div>
        """, unsafe_allow_html=True)

        history = get_chat_history(st.session_state.username)
        if history:
            for chat in history:
                st.markdown(f"**Input:** {chat['input']}")
                st.markdown(f"**Response:** {chat['response']}")
                st.markdown("---")
        else:
            st.write("No chat history available.")
