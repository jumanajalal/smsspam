import re
import unicodedata
def normalize_text(text):
    # Lowercase
    text = text.lower()

    # Remove excessive spacing (w i n → win)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\s+(?=\1)", r"\1", text)

    # Replace common leetspeak
    replacements = {
        "@": "a",
        "3": "e",
        "1": "i",
        "0": "o",
        "$": "s"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # Remove weird unicode accents
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    return text