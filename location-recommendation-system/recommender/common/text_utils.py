import re

TURKISH_SUFFIXES = {
    "a", "e", "ı", "i", "u", "ü", "y", "ya", "ye", "yi", "yu", "yü",
    "da", "de", "ta", "te", "dan", "den", "tan", "ten", "la", "le", "yla", "yle",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün",
    "sı", "si", "su", "sü",
    "lar", "ler", "larda", "lerde", "ların", "lerin", "ları", "leri",
    "ınav", "iniz", "imiz", "umuzu", "ümüzü", "ımızı", "imizi",
}

_EXCLUDED_STEMS = {"yol": "yolcu", "gar": "garip", "liman": "limon"}


def turkish_lower(text):
    if not text:
        return ""
    return (str(text).replace("İ", "i").replace("I", "ı").replace("Ğ", "ğ")
            .replace("Ü", "ü").replace("Ş", "ş").replace("Ö", "ö")
            .replace("Ç", "ç").lower())


def words(text):
    return re.findall(r"\w+", turkish_lower(text))


def turkish_word_match(text, keyword):
    if not text or not keyword:
        return False
    text_norm = turkish_lower(text)
    kw = turkish_lower(keyword)
    if kw not in text_norm:
        return False
    if " " in kw:
        return True
    for w in re.findall(r"\w+", text_norm):
        if w == kw:
            return True
        if w.startswith(kw):
            excluded = _EXCLUDED_STEMS.get(kw)
            if excluded and w.startswith(excluded):
                continue
            if w[len(kw):] in TURKISH_SUFFIXES:
                return True
    return False
