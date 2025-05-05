import os
import re
import random
import yaml
import logging
import torch
import torch.nn as nn
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize
from pyvi import ViTokenizer

# ---------- Text processing utils ----------
def clean_text(text: str) -> str:
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def hybrid_tokenize(text):
    ut = word_tokenize(text, format="text").split()
    pv = ViTokenizer.tokenize(text).split()
    merged, i, j = [], 0, 0
    while i < len(ut) and j < len(pv):
        if ut[i] == pv[j]:
            merged.append(ut[i]); i += 1; j += 1
        else:
            if len(ut[i]) >= len(pv[j]):
                merged.append(ut[i]); i += 1; j += len(pv[j].split('_'))
            else:
                merged.append(pv[j]); j += 1; i += len(ut[i].split('_'))
    merged += ut[i:] + pv[j:]
    return merged

# ---------- Load dialogue flow ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CONVERSATION_PATH = os.path.join(BASE_DIR, "src/bot/ConversationFlow.yaml")
with open(CONVERSATION_PATH, "r", encoding="utf-8") as f:
    DIALOGUE = yaml.safe_load(f)
print("Loaded states:", list(DIALOGUE.keys()))

# ---------- ML model for emotion detection ----------
MODEL_PATH = os.path.join(BASE_DIR, "src/nlp/phobert_sentiment_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhoBERTClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))

model = PhoBERTClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

LABEL_MAP = {
    0: "Vui_vẻ",
    1: "Tức_giận",
    2: "Buồn_bã",
    3: "Lo_lắng",
    4: "Trung_lập"
}

def predict_label_and_confidence(text: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    padding=True, max_length=256)
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()
    return LABEL_MAP[pred], confidence

def weighted_majority(emotions_with_confidence):
    score_map = {}
    for label, conf in emotions_with_confidence:
        score_map[label] = score_map.get(label, 0) + conf
    return max(score_map, key=score_map.get)

# ---------- Dialogue manager helpers ----------
def match_transition(trans: dict, user_text: str):
    txt = user_text.strip().lower()
    if "any" in trans:
        return "any"
    if "agree" in trans and re.search(r"\b(có|ok|đồng ý|sẵn sàng)\b", txt):
        return "agree"
    if "decline" in trans and re.search(r"\b(không|ko|không muốn)\b", txt):
        return "decline"
    for key in trans:
        norm = key.lower().replace("_", " ")
        if norm in txt:
            return key
    return None

def next_state(user_text: str, state: str, user_data: dict):
    cfg = DIALOGUE[state]["transitions"]
    choice = match_transition(cfg, user_text)
    if state == "ASK_READINESS_5" and choice == "agree":
        return "PROVIDE_EXERCISE"
    return cfg.get(choice)

# ---------- Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    greet = random.choice(DIALOGUE["START"]["bot"])
    await update.message.reply_text(greet)
    q1 = random.choice(DIALOGUE["ASK_FEEL_1"]["bot"])
    await update.message.reply_text(q1)
    context.user_data.clear()
    context.user_data["state"] = "ASK_FEEL_1"
    # Khởi tạo list để lưu (label, conf) 3 câu đầu
    context.user_data["responses"] = []

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    state = context.user_data.get("state")
    if state not in DIALOGUE:
        await start(update, context)
        return

    logging.info(f"[CHAT {update.effective_chat.id}] STATE={state} | USER={user_text}")

    # Với 3 lượt đầu tiên, predict và lưu vào responses
    if state in ["ASK_FEEL_1", "ASK_DETAIL_2", "ASK_COPING_3"]:
        label, conf = predict_label_and_confidence(user_text)
        context.user_data["responses"].append((label, conf))
        # Khi đã đủ 3 responses, tính nhãn cuối
        if len(context.user_data["responses"]) == 3:
            final_label = weighted_majority(context.user_data["responses"])
            context.user_data["label"] = final_label
            logging.info(f"[CHAT {update.effective_chat.id}] FINAL LABEL={final_label}")

    # Chuyển state tiếp theo
    nxt = next_state(user_text, state, context.user_data)
    if not nxt or nxt not in DIALOGUE:
        nxt = "END"

    logging.info(f"[CHAT {update.effective_chat.id}] → NEXT STATE={nxt}")

    # Soạn reply
    if nxt == "PROVIDE_EXERCISE":
        tpl = random.choice(DIALOGUE[nxt]["bot"])
        label = context.user_data.get("label", "Trung_lập")
        ex_text = DIALOGUE.get(f"EXERCISE_{label}", {}).get("exercise_text", "")
        reply = tpl.replace("{exercise_text}", ex_text)
    elif nxt.startswith("EXERCISE_") and "exercise_text" in DIALOGUE[nxt]:
        reply = DIALOGUE[nxt]["exercise_text"]
    else:
        reply = random.choice(DIALOGUE[nxt].get("bot", [""]))

    await update.message.reply_text(reply)
    context.user_data["state"] = nxt

# ---------- Main ----------
def main():
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("Missing TELEGRAM_TOKEN")

    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
