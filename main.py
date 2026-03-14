import os
from dotenv import load_dotenv  # Added for .env support
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from engine import verify_response 

# Load environment variables from a .env file if it exists
load_dotenv()

app = FastAPI(title="Real-time AI Hallucination Firewall")

# Fetch the key from the environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Check your .env file or terminal session.")

client = Groq(api_key=api_key)

class QueryRequest(BaseModel):
    user_query: str

@app.post("/firewall")
async def firewall_check(request: QueryRequest):
    # --- STEP A: REAL-TIME GENERATION ---
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant. Answer based ONLY on verified facts."},
            {"role": "user", "content": request.user_query}
        ],
        model="llama-3.1-8b-instant",
    )
    
    raw_ai_response = chat_completion.choices[0].message.content

    # --- STEP B: THE FIREWALL INTERCEPTION ---
    audit_logs = verify_response(raw_ai_response)

    # Generate the safe version for the user
    final_output = []
    for item in audit_logs:
        if "VERIFIED" in item["status"]: 
            final_output.append(item["sentence"])
        else:
            final_output.append("[REDACTED: Unsupported Claim]")
            
    firewalled_response = " ".join(final_output)
    
    return {
        "raw_ai_response": raw_ai_response,
        "firewalled_response": firewalled_response,
        "audit_logs": audit_logs
    }