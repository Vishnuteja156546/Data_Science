from dotenv import load_dotenv
import os

def load_env():
    load_dotenv()

def get_groq_key():
    return os.getenv("GROQ_API_KEY")

def get_groq_url():
    return os.getenv("GROQ_API_URL")
