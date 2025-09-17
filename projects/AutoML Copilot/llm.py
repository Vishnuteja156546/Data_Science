from openai import OpenAI
from utils import get_groq_key, get_groq_url

def call_grok(prompt, max_tokens=500, temperature=0.2):
    """
    Call Groq LLM API (OpenAI-compatible).
    """
    key = get_groq_key()
    url = get_groq_url()
    if not key or not url:
        raise ValueError("GROQ_API_KEY/GROQ_API_URL not set in .env")

    client = OpenAI(api_key=key, base_url=url)

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[
            {"role": "system", "content": "You are a helpful AI data science assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )

    return response.choices[0].message.content
