# agents/url_monitor_agent.py
import pandas as pd
import requests
import hashlib
import threading
import time
from logic.auto_pipeline import run_auto_pipeline

_monitored_urls = {}

def _download_and_hash(url):
    response = requests.get(url)
    response.raise_for_status()
    content = response.content
    return content, hashlib.md5(content).hexdigest()

def _monitor_url_loop(url, interval_seconds, model_mode):
    print(f"🌐 Monitoring URL: {url}")
    last_hash = None
    while True:
        try:
            content, new_hash = _download_and_hash(url)
            if new_hash != last_hash:
                print(f"📥 Change detected at {url}")
                df = pd.read_csv(pd.compat.StringIO(content.decode()))
                run_auto_pipeline(df, model_mode=model_mode)
                last_hash = new_hash
            else:
                print(f"⏳ No change at {url}")
        except Exception as e:
            print(f"❌ Error monitoring {url}: {e}")
        time.sleep(interval_seconds)

def start_url_monitoring(url, interval_seconds=60, model_mode="cloud"):
    if url in _monitored_urls:
        print(f"🔁 Already monitoring {url}")
        return
    thread = threading.Thread(
        target=_monitor_url_loop,
        args=(url, interval_seconds, model_mode),
        daemon=True
    )
    thread.start()
    _monitored_urls[url] = thread
