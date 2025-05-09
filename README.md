# Chatbot Phân Tích Cảm Xúc và Hỗ Trợ Cải Thiện Tâm Lý

## Giới thiệu
Dự án xây dựng một chatbot tiếng Việt có khả năng phân tích cảm xúc từ đoạn hội thoại và đưa ra phản hồi phù hợp nhằm hỗ trợ cải thiện sức khỏe tâm lý của người dùng. Chatbot có thể được triển khai trên Telegram.

## Mục tiêu
- Phân loại cảm xúc người dùng (Vui vẻ, Buồn bã, Tức giận, Lo lắng, Trung lập).
- Tương tác tự nhiên bằng ngôn ngữ tiếng Việt.
- Cung cấp phản hồi tâm lý học phù hợp với từng cảm xúc.
- Gợi ý các bài tập giúp cải thiện tinh thần (thiền, viết nhật ký, hít thở...).

## Các nhóm cảm xúc

| Nhóm cảm xúc        | Biểu hiện                            | Phản hồi gợi ý                                 |
|---------------------|--------------------------------------|------------------------------------------------|
| Vui vẻ / Hạnh phúc  | Hào hứng, tự tin                     | Khuyến khích duy trì thói quen tốt             |
| Buồn bã / Cô đơn    | Mất động lực, trống rỗng             | Gợi ý viết nhật ký, động viên tinh thần        |
| Tức giận / Khó chịu | Cáu giận, không hài lòng             | Hướng dẫn bài tập thở, kiểm soát cảm xúc       |
| Lo lắng / Sợ hãi    | Lo âu, bất an                        | Gợi ý thiền, thư giãn, hít thở sâu             |
| Trung lập           | Không có cảm xúc rõ ràng             | Giữ trò chuyện tự nhiên, khuyến khích tích cực |

## Công nghệ sử dụng
- **Ngôn ngữ**: Python 3.10+
- **Xử lý ngôn ngữ tự nhiên (NLP)**: [PhoBERT](https://huggingface.co/VinAI/phobert-base)
- **Phân loại cảm xúc**: Mô hình fine-tuned PhoBERT
- **Frontend**: Telegram Bot
- **Backend**: `python-telegram-bot`, Flask (tùy chọn mở rộng)
- **Thư viện chính**:
  - `transformers`, `torch`
  - `underthesea`, `pyvi`
  - `scikit-learn`, `pandas`, `numpy`
  - `python-dotenv`, `pyyaml`, `emoji`

## Cài đặt và chạy thử
```bash
git clone https://github.com/tronganh0104/EnPChatbot.git
cd EnPChatbot
pip install -r requirements.txt
```

- Tạo file .env ở thư mục src và thêm nội dung
```bash 
TELEGRAM_TOKEN=your_telegram_token_here
```
- Khởi động Chatbot
```bash
cd src/bot
python bot.py
```

## Kiến trúc hệ thống

```plaintext
Người dùng ↔ Chatbot ↔ Mô hình phân tích cảm xúc ↔ Phản hồi phù hợp
```
## Dataset
Dự án sử dụng các bộ dữ liệu sau để huấn luyện mô hình phân tích cảm xúc:

- **VMTEB (Vietnamese Multi-Task Emotion Benchmark)**: Bộ dữ liệu này cung cấp các nhãn cảm xúc cho văn bản tiếng Việt, bao gồm các cảm xúc như vui vẻ, buồn bã, tức giận, lo lắng và trung lập. Đây là bộ dữ liệu chính được sử dụng cho việc phân loại cảm xúc.
  
- **UIT-VSFC (University of Information Technology - Vietnamese Sentiment and Emotion Classification)**: Bộ dữ liệu này cũng chứa các nhãn cảm xúc và tình cảm từ các đoạn văn bản tiếng Việt, giúp làm phong phú thêm dữ liệu huấn luyện.

- **Dữ liệu sinh thêm**: Ngoài hai bộ dữ liệu chính trên, dữ liệu bổ sung đã được sinh ra từ các mô hình học máy và phương pháp tổng hợp để tạo ra nhiều tình huống đa dạng hơn, từ đó nâng cao độ chính xác của mô hình phân loại cảm xúc.

Các bộ dữ liệu này đã được tiền xử lý, bao gồm việc loại bỏ ký tự đặc biệt, chuẩn hóa văn bản tiếng Việt, và phân chia dữ liệu thành các tập huấn luyện và kiểm tra.

## Model
Fine-tune PhoBERT-base với 5 lớp cảm xúc.

## Giới hạn
- Chỉ phân tích văn bản (chưa hỗ trợ giọng nói hoặc hình ảnh).
- Không thể thay thế chuyên gia tâm lý học.