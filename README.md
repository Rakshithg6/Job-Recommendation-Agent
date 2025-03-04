# JobSync AI – Find Your Dream Job

## Overview
**JobSync AI** is a powerful tool that helps users find job opportunities and interact with AI for job-related queries. It integrates a **FastAPI backend** with a **Streamlit frontend** for a seamless experience.

## Features
- 🔍 **Job Search**: Enter a job query to find relevant job listings.  
- 💬 **AI Chatbot**: Ask questions about job opportunities and receive instant responses.  
- ⚡ **FastAPI Backend**: Handles job search and AI interactions.  
- 🎨 **Streamlit UI**: Provides a user-friendly interface.  

---

## Project Structure
```plaintext
📂 JobSync-AI/
│── 📜 fastapi_server.py   # FastAPI backend to handle job queries
│── 📜 streamlit_app.py    # Streamlit frontend for user interaction
│── 📜 README.md           # Project documentation
```

## Git Clone 
```
git clone https://github.com/your-username/JobSync-AI.git
cd JobSync-AI
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Run FastAPI Server
```
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

## Run Streamlit App
```
streamlit run streamlit_app.py
```



