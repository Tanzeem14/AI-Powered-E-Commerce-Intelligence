from google import genai
from dotenv import load_dotenv
import os
load_dotenv()
# client=genai.Client(api_key="AIzaSyC_aj20xs-OtUkyAHJiUaFgzbDO1U1OW-Q")
client=genai.Client(api_key=os.getenv("GROQ_API_KEY"))
models=client.models.list()
for m in models:
    print(m.name)