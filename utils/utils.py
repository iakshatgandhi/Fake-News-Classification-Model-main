from datetime import datetime

def save_feedback(text, model_prediction, user_correction, reason):
    feedback_data = {
        "text": text,
        "model_prediction": model_prediction,
        "user_correction": user_correction,
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }
    # In a real scenario, you would append this to a database or file
    print(f"Feedback saved: {feedback_data}")
