import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
for m in Groq(api_key=os.environ["GROQ_API_KEY"]).models.list().data:
    print(m.id)