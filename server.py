import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="SurakkhaShield BD API", version="2.0.0")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SmsRequest(BaseModel):
    sender: Optional[str] = "UNKNOWN"
    message: str

def analyze_sms_logic(sender: str, message: str):
    msg_lower = message.lower()
    sender_upper = sender.upper()
    
    score = 0
    indicators = []
    
    # 1. Official Whitelist Check (e.g., Grameenphone, Robi, bKash)
    official_senders = ["BKASH", "NAGAD", "ROCKET", "UPAY", "GP", "ROBI", "BL", "TELETALK", "16247", "16167", "16216"]
    if sender_upper in official_senders:
        if any(dom in msg_lower for dom in ["mygp.li", "robi.com.bd", "bkash.com", "nagad.com.bd"]):
            return {
                "score": 0,
                "risk_level": "SAFE / VERIFIED",
                "badge_class": "badge-safe",
                "summary": "Verified official transaction or operator notification.",
                "indicators": ["Official Sender ID", "Trusted Domain / Shortcode Match"]
            }

    # 2. Fake AI Police / Court Mamla / Warrant Scam
    mamla_keywords = ["মামলা", "ওয়ারেন্ট", "গ্রেফতার", "আদালত", "থানা", "dmp", "rab", "cid", "cyber tribunal", "warrant"]
    if any(kw in msg_lower for kw in mamla_keywords):
        score += 55
        indicators.append("AI / Cyber Police Case or Arrest Warrant Threat detected")
        
    # 3. MFS & OTP / PIN Harvesting Fraud
    otp_keywords = ["pin", "otp", "block", "suspend", "bhangabe", "bhangen", "reset", "verify", "অ্যাকাউন্ট সাময়িক বন্ধ"]
    if any(kw in msg_lower for kw in otp_keywords):
        score += 40
        indicators.append("MFS Account Suspend / PIN / OTP harvesting trap")
        
    # 4. Fake Cash-In Reversal Scam
    reversal_keywords = ["ভুলে টাকা চলে গেছে", "টাকা পাঠিয়েছি", "taka chole gece", "wrong send", "ferot din", "রিভার্স"]
    if any(kw in msg_lower for kw in reversal_keywords):
        score += 45
        indicators.append("Fake 'Wrong Cash-In Reversal' scam pattern detected")

    # 5. Malicious APK / Phishing Link Check
    if ".apk" in msg_lower or "app.apk" in msg_lower or "download" in msg_lower:
        score += 50
        indicators.append("Dangerous .apk application download link detected")
        
    if "http://" in msg_lower or "https://" in msg_lower:
        if not any(trusted in msg_lower for trusted in [".gov.bd", "bkash.com", "nagad.com.bd", "mygp.li"]):
            score += 25
            indicators.append("Unverified external tracking or phishing link found")

    # Personal Sender ID check for financial / legal claims
    digits = "".join(filter(str.isdigit, sender))
    if len(digits) >= 10:
        if digits.startswith("880") and len(digits) == 13:
            if not sender_upper in official_senders:
                score += 35
                indicators.append(f"Suspicious personal mobile number ({sender}) claiming official authority")

    if score > 100:
        score = 100

    if score >= 60:
        risk_level = "HIGH FRAUD RISK"
        badge_class = "badge-danger"
        summary = "CRITICAL WARNING: This message exhibits high probability of being a Bangladesh banking, MFS or AI Extortion scam!"
    elif score >= 25:
        risk_level = "SUSPICIOUS / CAUTION"
        badge_class = "badge-warning"
        summary = "WARNING: Potential deceptive pattern detected. Do not click links or share PIN/OTP."
    else:
        risk_level = "LOW RISK / LIKELY SAFE"
        badge_class = "badge-safe"
        summary = "No immediate high-risk fraud patterns detected."

    return {
        "score": score,
        "risk_level": risk_level,
        "badge_class": badge_class,
        "summary": summary,
        "indicators": indicators if indicators else ["Normal text structure, no suspicious keywords found."]
    }

@app.post("/api/v1/scan-message")
def scan_message(req: SmsRequest):
    result = analyze_sms_logic(req.sender, req.message)
    return JSONResponse(content=result)

@app.get("/api/v1/health")
def health_check():
    return {"status": "online", "engine": "Bangladesh Cyber & Banking Fraud Shield Active"}

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "project.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <head><title>SurakkhaShield BD</title></head>
        <body style="font-family:sans-serif; text-align:center; padding-top:50px;">
            <h2>SurakkhaShield BD Backend is Running!</h2>
            <p>Please place your <b>project.html</b> file in the same folder as server.py.</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)