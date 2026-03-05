# predict.py Spam and Malicious URL Detector
import re
import pickle
import pandas as pd
from urllib.parse import urlparse

# Load models
with open("models/spam_model.pkl", "rb") as f:
    spam_model = pickle.load(f)

with open("models/spam_tfidf.pkl", "rb") as f:
    spam_tfidf = pickle.load(f)

with open("models/url_model.pkl", "rb") as f:
    url_model = pickle.load(f)

with open("models/url_feature_cols.pkl", "rb") as f:
    url_feature_cols = pickle.load(f)

URL_LABEL_MAP = {0: "benign", 1: "defacement", 2: "phishing", 3: "malware"}
DANGEROUS_URL_TYPES = {"defacement", "phishing", "malware"}

# Text cleaning


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# URL extraction


def extract_urls(text):
    pattern = re.compile(
        r"(?:https?://|www\.)\S+"
        r"|(?:[a-zA-Z0-9\-]+\.)"
        r"(?:com|net|org|xyz|io|gov|edu"
        r"|info|co|me|tv|online|site"
        r"|web|app|dev|ai|click|link"
        r"|tk|ml|ga|cf|gq)\S*",
        re.IGNORECASE
    )
    return pattern.findall(text)

# URL feature extraction


def has_ip_address(url):
    return int(bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", url)))


def extract_url_features(url):
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        path, query = parsed.path, parsed.query
    except:
        domain, path, query = "", "", ""

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r"[=?&@%#_~]", url)),
        "num_subdomains": max(0, domain.count(".") - 1),
        "has_https": int(url.startswith("https")),
        "has_ip": has_ip_address(url),
        "has_at_symbol": int("@" in url),
        "has_double_slash": int("//" in url[7:]),
        "query_length": len(query),
        "num_query_params": len(query.split("&")) if query else 0,
    }
    return pd.DataFrame([[features[c] for c in url_feature_cols]],
                        columns=url_feature_cols)


# Main
if __name__ == "__main__":
    print("\nSpam & Malicious URL Detector")
    print("   Type 'quit' to exit\n")

    while True:
        text = input("Enter message: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not text:
            continue

        # 1 Spam check
        vec = spam_tfidf.transform([clean_text(text)])
        spam_label = spam_model.predict(vec)[0]
        spam_conf = spam_model.predict_proba(vec)[0][spam_label]
        spam_str = "SPAM" if spam_label == 1 else "HAM"

        # 2 URL check
        urls = extract_urls(text)

        # 3 Verdict
        has_danger_url = False
        url_results = []
        for url in urls:
            feat = extract_url_features(url)
            url_label = url_model.predict(feat)[0]
            url_conf = url_model.predict_proba(feat)[0][url_label]
            url_str = URL_LABEL_MAP[url_label]
            dangerous = url_str in DANGEROUS_URL_TYPES
            if dangerous:
                has_danger_url = True
            url_results.append((url, url_str, url_conf, dangerous))

        if spam_label == 1 and has_danger_url:
            verdict = "DANGEROUS"
        elif spam_label == 1 or has_danger_url:
            verdict = "SUSPICIOUS"
        else:
            verdict = "SAFE"

        # 4 Display results
        print(f"  MESSAGE : {spam_str} ({spam_conf*100:.1f}%)")
        if url_results:
            for url, label, conf, danger in url_results:
                flag = "SUS " if danger else "SAFE "
                print(
                    f"  URL     : {flag}{url} --> {label.upper()} ({conf*100:.1f}%)")
        else:
            print(f"  URL     : No URL detected")
        print(f"  VERDICT : {verdict}")
