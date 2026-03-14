import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from engine import verify_response

# Load environment variables
load_dotenv()

app = FastAPI(title="Real-time AI Hallucination Firewall")

# Setup Groq Client
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=api_key)


class QueryRequest(BaseModel):
    user_query: str


@app.post("/firewall")
async def firewall_check(request: QueryRequest):

    # STEP A — Generate response
    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical assistant. Answer based only on verified facts."
            },
            {
                "role": "user",
                "content": request.user_query
            }
        ]
    )

    raw_ai_response = chat_completion.choices[0].message.content

    # STEP B — Verify response
    audit_logs = verify_response(raw_ai_response)

    # STEP C — Build filtered output
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