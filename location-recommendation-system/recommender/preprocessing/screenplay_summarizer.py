import json
import os

SYSTEM_PROMPT = """Sen uzman bir Mekan Sorumlusu (Location Scout) ve Sinema Dramaturgusun.
Görevin: Verilen senaryoyu analiz edip genel senaryo havasını ve dış mekanlarını özetlemektir.

JSON FORMAT (Sadece bu şemada saf JSON döndür, açıklama veya ek yazı yazma):
{
  "genel_ozet": "Senaryonun genel akışını, karakterlerin dış mekanlardaki yolculuğunu ve geçtikleri mekanların sinematik atmosferini detaylı bir şekilde anlatan EN AZ 3-4 paragraf, 300-500 kelimelik geniş ve zengin bir özet. İlk paragraf genel konuyu ve ana hikayeyi, ikinci paragraf öne çıkan dış mekanları ve bu mekanların dramatik etkisini, üçüncü paragraf atmosfer, renk tonu ve sinematik yapıyı, dördüncü paragraf ise prodüksiyon/çekim lokasyonu önerilerini içermelidir."
}
"""

USER_INSTRUCTION = (
    "Aşağıdaki senaryo metnini temel alarak, yapımcı ve yönetmenler için sinematik dille "
    "yazılmış detaylı bir senaryo ve mekan özeti hazırla. 'genel_ozet' alanı en az 3-4 "
    "paragraftan oluşmalı ve 300-500 kelime uzunluğunda olmalıdır.\n\nSenaryo:\n"
)


def summarize_screenplay(text, model="llama-3.3-70b-versatile", temperature=0.5,
                         max_tokens=3000, max_chars=25000, api_key=None):
    from groq import Groq

    client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": USER_INSTRUCTION + text[:max_chars]}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    try:
        return json.loads(content).get("genel_ozet", content)
    except json.JSONDecodeError:
        return content
