## MedOracle: AI-Powered Multilingual Medical Chatbot

MedOracle is an AI-powered multilingual medical chatbot built using Python, Streamlit, and Machine Learning (KNN) for real-time symptom analysis, disease prediction, severity assessment, and personalized doctor recommendations. It integrates Natural Language Processing (SpaCy), MongoDB, and intelligent translation and speech modules to provide secure, accurate, and accessible healthcare guidance. 

The platform bridges the gap between users and reliable healthcare by offering an interactive interface that supports voice input, multilingual translation, and chat history tracking. With its combination of AI-driven prediction and user-friendly design, MedOracle enables individuals to perform quick self-assessments and obtain relevant medical guidance anytime, anywhere.

---

## Overview

MedOracle leverages Natural Language Processing (SpaCy), Machine Learning, and Speech & Translation APIs to analyze user symptoms, predict diseases, and provide immediate recommendations. It includes multilingual support, voice input, and text-to-speech features, making it accessible and user-friendly for diverse users.

---

## Features

- AI-powered disease prediction using KNN  
- Symptom severity analysis and prioritization  
- Real-time doctor and hospital recommendations  
- Voice input and text-to-speech integration  
- Multilingual support via Google Translator API  
- Secure login and signup using MongoDB and bcrypt  
- Chat history storage and retrieval  
- Interactive Streamlit web interface

---

## How It Works

- User inputs symptoms manually or through voice.
- NLP processes the text using SpaCy and TF-IDF vectorization.
- KNN classifier predicts the most likely disease.
- The system displays disease details, severity, and recommended doctors or hospitals.
- All user interactions are securely stored in MongoDB for future reference.

## Example Output

| Input                 | Predicted Disease | Severity | Doctor         |
| --------------------- | ----------------- | -------- | -------------- |
| Fever and sore throat | Tonsillitis       | Medium   | ENT Specialist |
| Chest pain            | Angina            | High     | Cardiologist   |
| Headache and nausea   | Migraine          | Low      | Neurologist    |

## Future Enhancements

- Integration with telemedicine APIs for live consultations
- Medication reminders and wearable device support
- Expanded multilingual support
- Cloud deployment on AWS or Azure
- Advanced disease models with deep learning
