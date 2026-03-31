import os
import sys
from dotenv import load_dotenv
load_dotenv('d:/Projects/AnomolyIQ/anomalyiq/backend/.env')
from openai import OpenAI
client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.getenv('OPENROUTER_API_KEY'))
try:
    response = client.chat.completions.create(
        model='nvidia/nemotron-3-super-120b-a12b:free',
        messages=[{'role': 'user', 'content': 'Test hello'}],
        max_tokens=50,
    )
    print("SUCCESS")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"ERROR: {type(e).__name__} - {str(e)}")
