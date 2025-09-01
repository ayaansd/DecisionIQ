# slack_utils.py
import requests

def send_summary_to_slack(summary_text: str, webhook_url: str, title: str = "📊 InsightGPT Summary"):
    if not webhook_url:
        return "Slack webhook URL not provided."
    
    payload = {
        "text": f"*{title}*\n{summary_text}"
    }

    response = requests.post(webhook_url, json=payload)

    if response.status_code == 200:
        return "✅ Summary sent to Slack successfully!"
    else:
        return f"⚠️ Failed to send to Slack. Status: {response.status_code}"
