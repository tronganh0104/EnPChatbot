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

def hybrid_tokenize(text: str):
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

def predict_label(text: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    padding=True, max_length=256)
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
    return LABEL_MAP[pred]

# ---------- Dialogue manager helpers ----------
def match_transition(trans: dict, user_text: str):
    txt = user_text.strip().lower()
    # 1) Ưu tiên any
    if "any" in trans:
        return "any"
    # 2) Agree / decline
    if "agree" in trans and re.search(r"\b(có|ok|đồng ý|sẵn sàng)\b", txt):
        return "agree"
    if "decline" in trans and re.search(r"\b(không|ko|không muốn)\b", txt):
        return "decline"
    # 3) Check explicit keys (ví dụ CLARIFY labels)
    for key in trans:
        norm = key.lower().replace("_", " ")
        if norm in txt:
            return key
    # 4) Không khớp
    return None

def next_state(user_text: str, state: str, user_data: dict):
    cfg = DIALOGUE[state]["transitions"]
    choice = match_transition(cfg, user_text)
    # Với ASK_READINESS_5 + agree → EXERCISE_{label}
    if state == "ASK_READINESS_5" and choice == "agree":
        label = user_data.get("label", "Trung_lập")
        return f"EXERCISE_{label}"
    return cfg.get(choice)

# ---------- Handlers ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Greeting
    greet = random.choice(DIALOGUE["START"]["bot"])
    await update.message.reply_text(greet)
    # Ngay lập tức hỏi lượt 1
    q1 = random.choice(DIALOGUE["ASK_FEEL_1"]["bot"])
    await update.message.reply_text(q1)
    # Reset user_data và set state về ASK_FEEL_1
    context.user_data.clear()
    context.user_data["state"] = "ASK_FEEL_1"

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    state = context.user_data.get("state")
    # Nếu state không hợp lệ, reset về start
    if state not in DIALOGUE:
        await start(update, context)
        return

    logging.info(f"[CHAT {update.effective_chat.id}] STATE={state} | USER={user_text}")

    # Lượt 1: detect emotion và lưu label
    if state == "ASK_FEEL_1":
        label = predict_label(user_text)
        context.user_data["label"] = label

    # Xác định next state
    nxt = next_state(user_text, state, context.user_data)
    if not nxt or nxt not in DIALOGUE:
        nxt = "END"

    logging.info(f"[CHAT {update.effective_chat.id}] → NEXT STATE={nxt}")

    # Chọn reply
    if nxt == "PROVIDE_EXERCISE":
        # Lấy template và thay placeholder
        template = random.choice(DIALOGUE[nxt]["bot"])
        label = context.user_data.get("label", "Trung_lập")
        ex_node = DIALOGUE.get(f"EXERCISE_{label}", {})
        exercise_text = ex_node.get("exercise_text", "")
        reply = template.replace("{exercise_text}", exercise_text)
    elif nxt.startswith("EXERCISE_") and "exercise_text" in DIALOGUE[nxt]:
        # Khi user yêu cầu hướng dẫn chi tiết
        reply = DIALOGUE[nxt]["exercise_text"]
    else:
        reply = random.choice(DIALOGUE[nxt].get("bot", [""]))

    # Gửi plain text và cập nhật state
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