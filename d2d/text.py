"""Language-specific tokenizers for the word-based methods (TF-IDF, LDA, NMF, CAt)."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# A short list of Arabic function words.
ARABIC_STOP = set("في من على إلى عن مع هذا هذه ذلك التي الذي كان كانت هو هي لا ما لم لن أن إن او أو ثم قد كل جدا و ف ب ل يا".split())


def get_tokenizer(lang):
    """Return a function that maps a text to a list of content words."""
    if lang == "th":
        # Thai has no spaces between words: PyThaiNLP newmm segmentation and its stop-word list.
        from pythainlp.corpus import thai_stopwords
        from pythainlp.tokenize import word_tokenize
        stop = set(thai_stopwords())

        def tok(t):
            return [w for w in word_tokenize(str(t).lower(), engine="newmm", keep_whitespace=False)
                    if w not in stop and len(w) > 1 and re.search(r"[฀-๿a-z]", w)]
        return tok
    if lang == "ar":
        def tok(t):
            t = re.sub(r"[ً-ْـ]", "", str(t))  # remove diacritics (harakat) and tatweel
            return [w for w in re.findall(r"[ء-ي]+", t) if w not in ARABIC_STOP and len(w) > 1]
        return tok

    def tok(t):
        return [w for w in re.findall(r"[a-z]+(?:'[a-z]+)?", str(t).lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return tok
