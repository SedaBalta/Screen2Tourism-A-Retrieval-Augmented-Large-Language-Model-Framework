from .text_utils import turkish_lower, turkish_word_match

CATEGORY_KEYWORDS = {
    "hastane_saglik": ["hastane", "klinik", "sağlık ocağı", "tıp merkezi", "eczane", "poliklinik", "dispanser", "şifa", "acil", "ambulans"],
    "is_endustri": ["fabrika", "plato", "depo", "sanayi", "atölye", "şantiye", "ofis", "holding", "dükkan", "mağaza", "hangar", "tersane", "depolar", "atölyeler"],
    "park_doga": ["park", "bahçe", "koru", "arboretum", "botanik", "orman", "yayla", "mesire", "piknik", "vadi", "dağ", "tepe", "plato", "kamp", "kaya", "kayalık", "uçurum", "yamaç", "patika"],
    "gar_ulasim": ["gar", "istasyon", "viyadük", "köprü", "demiryol", "ray", "havalimanı", "havaalanı", "terminal", "otogar", "liman", "iskele", "rıhtım", "peron", "tünel", "otoban", "geçit", "raylar", "makas"],
    "sokak_kentsel": ["sokak", "cadde", "yol", "mahalle", "bulvar", "meydan", "arasta", "çarşı", "pazar", "geçit", "kaldırım", "ara sokak", "arnavut kaldırımı", "caddeler"],
    "konut_tarihi": ["konak", "köşk", "saray", "apartman", "apartıman", "villa", "ev", "rezidans", "yalı", "kale", "hisar", "şato", "kervansaray", "han", "tapınak", "manastır", "kilise", "cami", "türbe", "yalılar", "konaklar"],
    "su_kenari": ["deniz", "sahil", "kıyı", "kumsal", "plaj", "göl", "baraj", "şelale", "nehir", "çay", "dere", "azmak", "kanyon", "koy", "körfez", "delta", "marina", "liman", "rıhtım", "iskele"],
    "sosyal_eglence": ["kafe", "cafe", "restoran", "lokanta", "bar", "kulüp", "meyhane", "otel", "pansiyon", "sinema", "tiyatro", "müze", "kütüphane", "sahaf", "kıraathane", "pavyon", "kulübü", "lokantası"],
    "resmi_kurum": ["karakol", "emniyet", "adliye", "meclis", "valilik", "belediye", "okul", "üniversite", "kampüs", "askeriye", "kışla", "cezaevi", "hapishane", "nezarethane", "adliyesi", "okulu"],
    "koy_kirsal": ["köy", "tarla", "samanlık", "ahır", "traktör", "çiftlik", "mera", "koyun", "keçi", "toprak yol", "kerpiç", "bağ", "bahçe", "ekin"],
    "tarihi_antik": ["antik", "harabe", "kalıntı", "sütun", "tapınak", "manastır", "antik kent", "sit alanı", "kazı", "lahit", "tiyatro", "surlar", "kule"],
    "ibadethane_inanc": ["cami", "kilise", "tapınak", "manastır", "sinagog", "havra", "türbe", "mescit", "mihrap", "minare", "çan kulesi"],
    "spor_rekreasyon": ["stadyum", "saha", "kort", "havuz", "spor salonu", "pist", "hipodrom", "antrenman", "tribün"],
    "askeri_guvenlik": ["kışla", "askeri", "karargah", "nöbet kulübesi", "cezaevi", "hapishane", "nezarethane", "üs bölgesi", "radar", "tel örgüler"],
}

CATEGORY_LABELS = {
    "hastane_saglik": "Sağlık & Hastane",
    "is_endustri": "İş & Endüstri",
    "park_doga": "Doğa & Orman",
    "gar_ulasim": "Ulaşım & Liman",
    "sokak_kentsel": "Sokak & Kentsel",
    "konut_tarihi": "Tarihi & Konut",
    "su_kenari": "Sahil & Su Kenarı",
    "sosyal_eglence": "Sosyal & Eğlence",
    "resmi_kurum": "Resmi & Eğitim",
    "koy_kirsal": "Köy & Kırsal",
    "tarihi_antik": "Tarihi Harabe & Antik Sit",
    "ibadethane_inanc": "İbadethane & İnanç",
    "spor_rekreasyon": "Spor & Rekreasyon",
    "askeri_guvenlik": "Askeri & Güvenlik",
}

GENERAL_CATEGORY_LABEL = "Genel Dış Mekan"

GENRE_ADJUSTMENTS = {
    "Polisiye": {
        "boost": ["is_endustri", "sokak_kentsel", "resmi_kurum", "askeri_guvenlik"],
        "penalty": ["su_kenari", "sosyal_eglence", "koy_kirsal"],
    },
    "Aksiyon": {
        "boost": ["is_endustri", "gar_ulasim", "sokak_kentsel", "askeri_guvenlik"],
        "penalty": ["sosyal_eglence"],
    },
    "Romantik": {
        "boost": ["su_kenari", "sosyal_eglence", "park_doga", "koy_kirsal"],
        "penalty": ["is_endustri", "resmi_kurum", "askeri_guvenlik"],
    },
    "Komedi": {
        "boost": ["sokak_kentsel", "sosyal_eglence", "konut_tarihi"],
        "penalty": ["askeri_guvenlik", "tarihi_antik"],
    },
    "Dram": {
        "boost": ["konut_tarihi", "sokak_kentsel", "park_doga", "koy_kirsal", "ibadethane_inanc"],
        "penalty": [],
    },
    "Fantastik": {
        "boost": ["tarihi_antik", "park_doga", "konut_tarihi", "ibadethane_inanc"],
        "penalty": ["is_endustri", "resmi_kurum"],
    },
}

REGIONS = [
    (("ege",), ["İzmir", "Muğla", "Aydın", "Denizli", "Manisa", "Afyonkarahisar", "Kütahya", "Uşak", "Balıkesir"]),
    (("karadeniz",), ["Trabzon", "Rize", "Artvin", "Giresun", "Ordu", "Samsun", "Sinop", "Zonguldak", "Bolu", "Kastamonu", "Amasya", "Tokat", "Çorum", "Karabük", "Bartın", "Düzce", "Gümüşhane", "Bayburt"]),
    (("akdeniz",), ["Antalya", "Mersin", "Adana", "Hatay", "Kahramanmaraş", "Osmaniye", "Isparta", "Burdur"]),
    (("marmara",), ["İstanbul", "Bursa", "Kocaeli", "Balıkesir", "Çanakkale", "Edirne", "Tekirdağ", "Kırklareli", "Yalova", "Sakarya", "Bilecik"]),
    (("güneydoğu", "guneydogu"), ["Mardin", "Gaziantep", "Şanlıurfa", "Diyarbakır", "Batman", "Siirt", "Şırnak", "Kilis", "Adıyaman"]),
    (("doğu", "dogu"), ["Erzurum", "Van", "Malatya", "Elazığ", "Kars", "Ağrı", "Iğdır", "Ardahan", "Muş", "Bitlis", "Hakkari", "Bingöl", "Tunceli", "Erzincan"]),
    (("anadolu", "orta anadolu", "iç anadolu", "ic anadolu"), ["Ankara", "Konya", "Kayseri", "Eskişehir", "Sivas", "Kırıkkale", "Kırşehir", "Yozgat", "Nevşehir", "Niğde", "Aksaray", "Karaman", "Çankırı"]),
]


def location_category(text):
    for key, keywords in CATEGORY_KEYWORDS.items():
        if any(turkish_word_match(text, kw) for kw in keywords):
            return key
    return None


def category_label(text):
    key = location_category(text)
    return CATEGORY_LABELS[key] if key else GENERAL_CATEGORY_LABEL


def active_categories(text):
    return {key for key, keywords in CATEGORY_KEYWORDS.items()
            if any(turkish_word_match(text, kw) for kw in keywords)}


def region_provinces(text):
    t = turkish_lower(text)
    for names, provinces in REGIONS:
        if any(n in t for n in names):
            return provinces
    return []
