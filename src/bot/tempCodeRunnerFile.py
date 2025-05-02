suggestion = SUGGESTIONS[label]
    reply = (
        f"→ Mình nhận thấy bạn đang *{label}* (độ tin cậy {conf:.2f}).\n"
        f"Gợi ý hỗ trợ: {suggestion}"
    )
    await update.message.reply_markdown(reply)