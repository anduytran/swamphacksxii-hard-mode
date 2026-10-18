"""The languages a household can choose for the wearer's spoken answers.

A household's `language` (BCP-47, default "en") decides what language the answer, the
fixed phrases and the dashboard summary are in. Gemini transcribes, embeds and speaks all
of these; on-device Moonshine is English-only (see device/README.md).

Keep LANGUAGES in step with web/src/lib/languages.ts.
"""

DEFAULT_LANGUAGE = "en"

# code -> (English name, used in prompts; own name, shown on the dashboard)
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("English", "English"),
    "es": ("Spanish", "Español"),
    "zh": ("Mandarin Chinese", "中文"),
    "vi": ("Vietnamese", "Tiếng Việt"),
    "tl": ("Tagalog", "Tagalog"),
    "fr": ("French", "Français"),
    "ar": ("Arabic", "العربية"),
    "ko": ("Korean", "한국어"),
    "ru": ("Russian", "Русский"),
    "pt": ("Portuguese", "Português"),
    "hi": ("Hindi", "हिन्दी"),
    "de": ("German", "Deutsch"),
    "ht": ("Haitian Creole", "Kreyòl ayisyen"),
}


def normalize(code: str | None) -> str:
    """A known language code, or the default: "ES" -> "es", None / "xx" -> "en"."""
    code = (code or "").strip().lower()
    return code if code in LANGUAGES else DEFAULT_LANGUAGE


def is_english(code: str | None) -> bool:
    return normalize(code) == "en"


def language_name(code: str | None) -> str:
    return LANGUAGES[normalize(code)][0]


def instruction(code: str | None) -> str:
    """The line added to a system prompt so the model writes in the household's language."""
    return f"Write your reply in {language_name(code)}, whatever language the question or notes are in."


# Fixed spoken replies (keyed by the English text in ai.py), translated so the wearer hears
# them in their language. Each translation is itself a fixed phrase, so its audio is cached too.
_PHRASES: dict[str, dict[str, str]] = {
    "Sorry, I didn't catch that. Could you ask again?": {
        "es": "Perdón, no le entendí. ¿Puede preguntar otra vez?",
        "zh": "对不起，我没听清楚。您能再问一遍吗？",
        "vi": "Xin lỗi, tôi nghe không rõ. Bạn có thể hỏi lại được không?",
        "tl": "Pasensya na, hindi ko po narinig. Puwede po bang ulitin ang tanong?",
        "fr": "Désolé, je n'ai pas bien compris. Pouvez-vous redemander ?",
        "ar": "عذرًا، لم أفهم ذلك. هل يمكنك أن تسأل مرة أخرى؟",
        "ko": "죄송해요, 잘 못 들었어요. 다시 물어봐 주시겠어요?",
        "ru": "Извините, я не расслышал. Можете спросить ещё раз?",
        "pt": "Desculpe, não entendi. Pode perguntar de novo?",
        "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। क्या आप फिर से पूछ सकते हैं?",
        "de": "Entschuldigung, das habe ich nicht verstanden. Können Sie noch einmal fragen?",
        "ht": "Padon, mwen pa t tande sa. Èske ou ka mande ankò?",
    },
    "Sorry, I can't help with that one.": {
        "es": "Lo siento, con eso no puedo ayudarle.",
        "zh": "对不起，这个我帮不了您。",
        "vi": "Xin lỗi, tôi không giúp được việc này.",
        "tl": "Pasensya na po, hindi ko kayo matutulungan diyan.",
        "fr": "Désolé, je ne peux pas vous aider pour cela.",
        "ar": "عذرًا، لا أستطيع المساعدة في هذا.",
        "ko": "죄송해요, 그건 도와드릴 수 없어요.",
        "ru": "Извините, с этим я помочь не могу.",
        "pt": "Desculpe, não posso ajudar com isso.",
        "hi": "माफ़ कीजिए, इसमें मैं मदद नहीं कर सकता।",
        "de": "Entschuldigung, dabei kann ich nicht helfen.",
        "ht": "Padon, mwen pa ka ede w ak sa.",
    },
    "I'm getting a lot of questions right now. Please ask me again in a minute.": {
        "es": "Estoy recibiendo muchas preguntas ahora mismo. Pregúnteme otra vez en un minuto, por favor.",
        "zh": "我现在收到的问题有点多。请过一分钟再问我。",
        "vi": "Lúc này tôi đang nhận quá nhiều câu hỏi. Vui lòng hỏi lại tôi sau một phút.",
        "tl": "Marami po akong tanong na natatanggap ngayon. Pakitanong po ulit pagkalipas ng isang minuto.",
        "fr": "Je reçois beaucoup de questions en ce moment. Redemandez-moi dans une minute, s'il vous plaît.",
        "ar": "تصلني أسئلة كثيرة الآن. من فضلك اسألني مرة أخرى بعد دقيقة.",
        "ko": "지금 질문이 많이 들어오고 있어요. 1분 뒤에 다시 물어봐 주세요.",
        "ru": "Сейчас мне задают много вопросов. Пожалуйста, спросите меня ещё раз через минуту.",
        "pt": "Estou recebendo muitas perguntas agora. Por favor, pergunte de novo daqui a um minuto.",
        "hi": "अभी मेरे पास बहुत सारे सवाल आ रहे हैं। कृपया एक मिनट बाद फिर से पूछिए।",
        "de": "Ich bekomme gerade sehr viele Fragen. Bitte fragen Sie mich in einer Minute noch einmal.",
        "ht": "M ap resevwa anpil kesyon kounye a. Tanpri mande m ankò nan yon minit.",
    },
    "Sorry, I couldn't come up with an answer just now.": {
        "es": "Lo siento, ahora mismo no encontré una respuesta.",
        "zh": "对不起，我现在想不出答案。",
        "vi": "Xin lỗi, lúc này tôi chưa tìm ra câu trả lời.",
        "tl": "Pasensya na po, wala akong maisagot ngayon.",
        "fr": "Désolé, je n'ai pas trouvé de réponse pour le moment.",
        "ar": "عذرًا، لم أتمكن من إيجاد إجابة الآن.",
        "ko": "죄송해요, 지금은 답을 찾지 못했어요.",
        "ru": "Извините, сейчас я не смог найти ответ.",
        "pt": "Desculpe, não consegui encontrar uma resposta agora.",
        "hi": "माफ़ कीजिए, अभी मुझे कोई जवाब नहीं मिला।",
        "de": "Entschuldigung, mir ist gerade keine Antwort eingefallen.",
        "ht": "Padon, mwen pa t jwenn yon repons kounye a.",
    },
}


def phrase(english: str, code: str | None) -> str:
    """`english` (a fixed phrase) in the household's language, or as is if there's no translation."""
    return _PHRASES.get(english, {}).get(normalize(code), english)


def all_translations(english: str) -> set[str]:
    return {english, *_PHRASES.get(english, {}).values()}
