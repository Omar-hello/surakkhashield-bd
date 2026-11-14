import os
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="SurakkhaShield BD ML API", version="3.0.0")

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
    use_ai: Optional[bool] = True

# --- MACHINE LEARNING DATASET LOADER & TRAINER ---
DATASET_PATH = os.path.join(os.path.dirname(__file__), "surokkhashield_1000_dataset.txt")

class SmsMLClassifier:
    def __init__(self, dataset_file: str):
        self.messages: List[str] = []
        self.labels: List[int] = []  # 1 = FAKE, 0 = REAL
        self.senders: List[str] = []
        self.raw_records: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="char_wb",
            min_df=1,
            max_features=8000
        )
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            min_df=1
        )
        self.nb_char = MultinomialNB(alpha=0.2)
        self.nb_word = MultinomialNB(alpha=0.2)
        self.tfidf_matrix = None
        self.is_trained = False
        self.load_and_train(dataset_file)

    def normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\b01[3-9]\d{8}\b", " <BD_PHONE> ", text)
        text = re.sub(r"\b\d{1,3}(,\d{3})*(\.\d{2})?\b", " <AMOUNT> ", text)
        text = re.sub(r"\b[a-z0-9]{8,12}\b", " <TRX_ID> ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def load_and_train(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"[WARNING] Dataset not found at {filepath}")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("["):
                    continue
                parts = [p.strip() for p in line.split("|", 3)]
                if len(parts) == 4:
                    sms_id, label_str, sender, body = parts
                    label = 1 if label_str.upper() == "FAKE" else 0
                    combined = f"{sender} {body}"
                    norm_text = self.normalize_text(combined)
                    self.messages.append(norm_text)
                    self.labels.append(label)
                    self.senders.append(sender)
                    self.raw_records.append({
                        "id": sms_id,
                        "label": label_str.upper(),
                        "sender": sender,
                        "body": body
                    })

        if len(self.messages) > 10:
            X_char = self.vectorizer.fit_transform(self.messages)
            X_word = self.word_vectorizer.fit_transform(self.messages)
            y = np.array(self.labels)
            self.nb_char.fit(X_char, y)
            self.nb_word.fit(X_word, y)
            self.tfidf_matrix = X_char
            self.is_trained = True
            print(f"[ML ENGINE] Trained successfully on {len(self.messages)} SMS")

    def predict(self, sender: str, message: str) -> Dict[str, Any]:
        if not self.is_trained:
            return {"trained": False, "fake_prob": 0.5, "similarity": 0.0, "closest_match": None}

        combined = f"{sender} {message}"
        norm_query = self.normalize_text(combined)

        X_q_char = self.vectorizer.transform([norm_query])
        X_q_word = self.word_vectorizer.transform([norm_query])

        prob_char = float(self.nb_char.predict_proba(X_q_char)[0][1])
        prob_word = float(self.nb_word.predict_proba(X_q_word)[0][1])

        sims = cosine_similarity(X_q_char, self.tfidf_matrix)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        closest_record = self.raw_records[best_idx]

        nn_fake_prob = 1.0 if closest_record["label"] == "FAKE" else 0.0
        if best_sim >= 0.45:
            blended_fake_prob = (0.45 * prob_char) + (0.35 * prob_word) + (0.20 * nn_fake_prob)
        else:
            blended_fake_prob = (0.55 * prob_char) + (0.45 * prob_word)

        return {
            "trained": True,
            "fake_prob": round(blended_fake_prob * 100, 1),
            "similarity": round(best_sim * 100, 1),
            "closest_match": closest_record
        }

ml_classifier = SmsMLClassifier(DATASET_PATH)

def analyze_sms_logic(sender: str, message: str):
    sender_clean = (sender or "UNKNOWN").strip()
    msg_lower = message.lower()
    sender_upper = sender_clean.upper().replace("-", "").replace(" ", "")

    ml_res = ml_classifier.predict(sender_clean, message)
    ml_fake_score = ml_res["fake_prob"] if ml_res["trained"] else 50.0
    similarity_pct = ml_res["similarity"]
    closest = ml_res["closest_match"]

    heuristic_score = 0
    indicators = []

    official_senders = {
        "BKASH", "NAGAD", "ROCKET", "UPAY", "CELLFIN", "DBBL", "IBBL", "CITYBANK",
        "BRACBANK", "EBL", "UCB", "MTB", "SCB", "GP", "ROBI", "BANGLALINK", "TELETALK",
        "16247", "16167", "16216", "16268", "16259", "16221", "16219"
    }

    fake_spoof_senders = {"BTRC-INFO", "CYBER-POLICE", "NAGAD-WIN", "BKASH-OFFER", "DBBL-ALERT", "BTRC-ALERT"}

    if sender_clean.upper() in fake_spoof_senders:
        heuristic_score += 65
        indicators.append({
            "severity": "high",
            "en": f"Spoofed Scam Sender ID ({sender_clean}): Official BTRC, Police & Banks never use '{sender_clean}'.",
            "bn": f"ভুয়া প্রতারক সেন্ডার আইডি ({sender_clean}): সরকারি সংস্থা বা ব্যাংক কখনোই এই নাম ব্যবহার করে না।"
        })

    digits = "".join(filter(str.isdigit, sender_clean))
    if (len(digits) == 11 and digits.startswith("01")) or (len(digits) == 13 and digits.startswith("8801")):
        heuristic_score += 45
        indicators.append({
            "severity": "high",
            "en": f"Personal Mobile Number Sender ({sender_clean}): Banks and Police never send official SMS from 01X numbers.",
            "bn": f"ব্যক্তিগত মোবাইল নম্বর ({sender_clean}): ব্যাংক বা পুলিশ কখনোই ব্যক্তিগত নম্বর থেকে মেসেজ পাঠায় না।"
        })

    if any(kw in msg_lower for kw in ["মামলা", "ওয়ারেন্ট", "গ্রেফতার", "আদালত", "ট্রাইব্যুনাল", "সাইবার ক্রাইম", "mamla", "warrant", "tribunal"]):
        heuristic_score += 60
        indicators.append({
            "severity": "high",
            "en": "Fake Cyber Police / Court Mamla Extortion Pattern detected.",
            "bn": "ভুয়া সাইবার পুলিশ বা ডিজিটাল মামলার ভয় দেখিয়ে প্রতারণার প্যাটার্ন শনাক্ত হয়েছে।"
        })

    if any(kw in msg_lower for kw in ["ভুল করে", "ভুলে আপনার নম্বরে", "ফেরত দিন", "হিসাব মিলছে না"]):
        heuristic_score += 55
        indicators.append({
            "severity": "high",
            "en": "Fake 'Wrong Cash-In / Agent Mistake' Reversal Scam detected.",
            "bn": "ভুয়া 'ভুল করে টাকা চলে গেছে / এজেন্ট ভুল' প্রতারণা শনাক্ত হয়েছে।"
        })

    if any(kw in msg_lower for kw in ["anydesk", "teamviewer", ".apk", ".xyz", ".top", ".info", "t.me/"]):
        heuristic_score += 60
        indicators.append({
            "severity": "high",
            "en": "Malicious APK, AnyDesk Remote Access, or Phishing Domain detected.",
            "bn": "ক্ষতিকর APK অ্যাপ, AnyDesk বা ফিশিং লিংক শনাক্ত হয়েছে।"
        })

    is_official_warning = any(safe_phrase in msg_lower for safe_phrase in [
        "never share", "do not share", "কখনোই কারো সাথে শেয়ার করবেন না"
    ])
    if not is_official_warning and any(kw in msg_lower for kw in ["কোডটি শেয়ার করুন", "কোডটি এই নম্বরে পাঠান", "পিন ও ওটিপি দিন"]):
        heuristic_score += 50
        indicators.append({
            "severity": "high",
            "en": "OTP / PIN Harvesting Trap: Message asks you to send or enter your secret PIN/OTP.",
            "bn": "পিন/ওটিপি চুরির ফাঁদ: মেসেজটিতে আপনার গোপন পিন বা ওটিপি চাওয়া হয়েছে।"
        })

    if sender_upper in official_senders and is_official_warning and heuristic_score == 0:
        final_score = int(min(15, round(ml_fake_score * 0.2)))
    elif sender_upper in official_senders and any(k in msg_lower for k in ["trxid", "avl bal", "available balance", "verification code is"]) and heuristic_score == 0:
        final_score = int(min(18, round(ml_fake_score * 0.25)))
    else:
        combined_score = (0.70 * ml_fake_score) + (0.30 * min(100, heuristic_score))
        if heuristic_score >= 55:
            combined_score = max(combined_score, 82.0)
        final_score = int(max(0, min(100, round(combined_score))))

    if ml_res["trained"] and closest:
        match_label = closest["label"]
        match_id = closest["id"]
        match_snippet = closest["body"][:85] + ("..." if len(closest["body"]) > 85 else "")
        indicators.insert(0, {
            "severity": "high" if match_label == "FAKE" and similarity_pct >= 30 else ("safe" if match_label == "REAL" and similarity_pct >= 30 else "medium"),
            "en": f"ML Dataset Match ({similarity_pct}% similar to {match_id} [{match_label}]): '{match_snippet}'",
            "bn": f"ML ডেটাসেট মিল ({match_id} [{match_label}]-এর সাথে {similarity_pct}% সাদৃশ্য): '{match_snippet}'"
        })

    if final_score >= 55:
        risk_level = "HIGH FRAUD RISK (FAKE)"
        badge_class = "badge-danger"
        summary = "DANGER: Machine Learning & Forensic Engine classified this SMS as FAKE / SCAM!"
    elif final_score >= 25:
        risk_level = "SUSPICIOUS / CAUTION"
        badge_class = "badge-warning"
        summary = "CAUTION: Suspicious pattern detected. Verify via official helpline."
    else:
        risk_level = "SAFE / VERIFIED (REAL)"
        badge_class = "badge-safe"
        summary = "VERIFIED: Machine Learning & Forensic Engine classified this SMS as REAL / SAFE."

    return {
        "score": final_score,
        "ml_probability": ml_fake_score,
        "dataset_similarity": similarity_pct,
        "closest_dataset_id": closest["id"] if closest else None,
        "closest_dataset_label": closest["label"] if closest else None,
        "engine": f"Scikit-Learn TF-IDF + Naive Bayes ({len(ml_classifier.messages)} SMS Dataset)",
        "risk_level": risk_level,
        "badge_class": badge_class,
        "summary": summary,
        "indicators": indicators
    }

@app.post("/api/v1/scan-message")
def scan_message(req: SmsRequest):
    result = analyze_sms_logic(req.sender or "UNKNOWN", req.message)
    return JSONResponse(content=result)

@app.get("/api/v1/dataset-stats")
def dataset_stats():
    return {
        "trained": ml_classifier.is_trained,
        "total_samples": len(ml_classifier.messages),
        "fake_samples": sum(ml_classifier.labels),
        "real_samples": len(ml_classifier.labels) - sum(ml_classifier.labels)
    }

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "ml_trained": ml_classifier.is_trained,
        "dataset_size": len(ml_classifier.messages),
        "engine": "SurakkhaShield BD ML Engine (TF-IDF + Naive Bayes + Similarity)"
    }

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "project.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h2>SurakkhaShield BD ML Backend Running!</h2>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
