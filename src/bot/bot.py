import re
import os
import torch
import logging
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from transformers import AutoModel, AutoTokenizer
from underthesea import word_tokenize
from pyvi import ViTokenizer
import torch.nn as nn

def clean_text(text):
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
            merged.append(ut[i]); i+=1; j+=1
        else:
            if len(ut[i]) >= len(pv[j]):
                merged.append(ut[i]); i+=1; j+=len(pv[j].split('_'))
            else:
                merged.append(pv[j]); j+=1; i+=len(ut[i].split('_'))
    merged += ut[i:] + pv[j:]
    return merged

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "src/nlp/phobert_sentiment_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["Vui vẻ", "Tức giận", "Buồn bã", "Sợ hãi", "Trung lập"]
SUGGESTIONS = {
    "Vui vẻ": "Tuyệt vời! Hãy duy trì những hoạt động khiến bạn cảm thấy hạnh phúc nhé 😊",
    "Tức giận": "Khi căng thẳng, bạn có thể thử bài tập thở: hít 4 giây, giữ 4 giây, thở ra 4 giây.",
    "Buồn bã": "Thử viết nhật ký: ghi lại những điều khiến bạn biết ơn hôm nay.",
    "Sợ hãi": "Hãy thử thiền ngắn 5 phút: ngồi yên, hít thở sâu và quan sát suy nghĩ.",
    "Trung lập": "Bạn có thể chia sẻ thêm hoặc thử tập trung vào điều tích cực xung quanh."
}

class PhoBERTClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(PhoBERTClassifier, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled_output))
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhoBERTClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

label_map = {
    0: "Vui vẻ",
    1: "Tức giận",
    2: "Buồn bã",
    3: "Sợ hãi",
    4: "Trung lập"
}
suggestions = {
    "Vui vẻ": "Thật tuyệt! Hãy tiếp tục giữ tinh thần tích cực nhé!",
    "Tức giận": "Hãy hít thở sâu và thư giãn. Viết ra cảm xúc có thể giúp bạn bình tĩnh hơn.",
    "Buồn bã": "Bạn có muốn chia sẻ thêm? Tôi luôn ở đây lắng nghe bạn.",
    "Sợ hãi": "Hãy tìm một nơi an toàn và nói chuyện với người bạn tin tưởng.",
    "Trung lập": "Bạn đang cảm thấy bình thường. Hãy thử làm điều gì đó bạn yêu thích nhé!"
}

# Dự đoán cảm xúc
def predict_and_suggest(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = label_map[pred]
    return label, confidence, suggestions[label]

# Handler lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào bạn! Tôi là chatbot phân tích cảm xúc. Hãy gửi một câu và tôi sẽ đoán cảm xúc cho bạn nhé."
    )

# Handler tin nhắn văn bản
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    label, conf, suggestion = predict_and_suggest(text)
    reply = f"→ Cảm xúc: *{label}* (độ tin cậy {conf:.2f})\n\nGợi ý: {suggestion}"
    await update.message.reply_markdown(reply)

# Hàm main
def main():
    TOKEN = "7617260436:AAEEMudZ1cf5LoGMcfKPRT3ep2y3RvRXe88"  # Thay bằng token của bạn
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot đang chạy...")
    app.run_polling()

if __name__ == "__main__":
    main()