"""Is a spoken clip a memory to save or a question to answer? For the wake-phrase route, where there's
no REMEMBER or ASK button to say so. Plain rules on the transcript: no model call, so free and instant.

    "I'm putting my keys on the counter"   -> remember
    "Where are my keys?"                   -> ask
    "Last Scene, did I take my pills?"     -> ask   (the wake phrase is stripped first)
"""

import re

# Wake phrases the transcript may start with (speech-to-text often keeps them).
# Speech-to-text often hears "Last Scene" as "last seen" or one word, so those count too.
WAKE_PHRASES = (
    "hey last scene", "hey lastscene", "hey last seen", "okay last scene", "ok last scene",
    "last scene", "lastscene", "last seen", "hey jarvis",
)

_QUESTION_START = re.compile(
    r"^(where|what|when|which|who|whose|how|why|did|do|does|have|has|had|is|are|was|were|can|could|would|will|"
    r"should|tell me|remind me|help me find|find|show me|any idea|got any idea)\b"
)
_QUESTION_ANYWHERE = re.compile(
    r"\b(where('s| is| are| was| were| did| do| have| has)\b|where did i\b|have you seen|did i (leave|put|take|lock|turn)\b|"
    r"do you know where\b|can you find\b|i can'?t find\b|i lost\b|i'?ve lost\b)"
)


def strip_wake_phrase(text: str) -> str:
    t = text.strip()
    low = re.sub(r"[^\w\s']", "", t.lower())
    for phrase in WAKE_PHRASES:
        if low.startswith(phrase):
            # drop the phrase plus any punctuation after it ("Hey Jarvis, where...")
            words = len(phrase.split())
            return re.sub(r"^\W*" + r"\W+".join([r"\w+"] * words) + r"[\W]*", "", t, count=1).strip()
    return t


# "Last Scene" as speech-to-text writes it, anywhere in the clip (the clip starts up to WAKE_PREROLL_SECONDS
# before the wake, so other talk can come first).
_WAKE_ANYWHERE = re.compile(
    r"\b(?:(?:hey|okay|ok)\W+)?(?:last\W*scenes?|last\W+seen|lost\W+scene|hey\W+jarvis)\b[\W_]*", re.IGNORECASE
)


def after_wake_phrase(transcript: str) -> str | None:
    """The words after "Last Scene" (the command), or None when the phrase isn't there at all: a false
    wake. Anything said before the phrase is dropped."""
    m = _WAKE_ANYWHERE.search(transcript or "")
    return None if m is None else transcript[m.end():].strip()


_CONTINUE = re.compile(r"^\W*(?:continue|go on|and also|follow(?:ing)? up)\b[\W_]*", re.IGNORECASE)


def continue_command(words: str) -> str | None:
    """'continue, what about my glasses' -> 'what about my glasses' (a follow-up in the same conversation);
    'continue' alone -> ''. None when it isn't a continue command."""
    m = _CONTINUE.match(words or "")
    return None if m is None else words[m.end():].strip()


# Questions that lean on the last answer ("point me in the right direction", "which way?", "is it still there?").
_NEEDS_CONTEXT = re.compile(
    r"\b(point me|right direction|which (way|direction|side)|where exactly|guide me|lead me|take me( there| to it)?|"
    r"walk me|how do i get (there|to (it|them))|(is|are) (it|they) still|what'?s (next to|near|around) (it|them)|"
    r"(left|right) or (left|right)|closer|turn (left|right))\b",
    re.IGNORECASE,
)


def needs_context(words: str) -> bool:
    """A follow-up that only makes sense after the last answer, said without "continue"."""
    return bool(_NEEDS_CONTEXT.search(words or ""))


def has_wake_phrase(transcript: str) -> bool:
    """Whether the words start with the wake phrase, allowing a word or two before it ("Okay, Last Scene").
    The wearer's own wake model can fire on other speech; speech-to-text then confirms it was said."""
    low = " ".join(re.sub(r"[^\w\s']", " ", transcript.lower()).split())
    return any(re.match(r"^(\w+ ){0,2}" + re.escape(p) + r"\b", low) for p in WAKE_PHRASES)


_FOUND = re.compile(r"\b(i('ve| have)? found|found (my|the|it|them)|got (my|the) [\w ]+ back)\b")


def match_item(text: str, names: dict[str, str]) -> str:
    """The saved item the words mention, by name: `names` maps every word people use (Teach names,
    aliases, items the camera or memories recorded) to the item's name. Longest match wins, so
    "reading glasses" beats "glasses"; "key" matches "keys" and "keys" matches "key". "" if none."""
    low = " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
    for said in sorted(names, key=len, reverse=True):
        words = said.lower().split()
        if not words:
            continue
        stem = words[-1][:-1] if words[-1].endswith("s") and len(words[-1]) > 3 else words[-1]
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in words[:-1] + [stem]) + r"(s|es)?\b"
        if re.search(pattern, low):
            return names[said]
    return ""


def fallback_command(transcript: str, names: dict[str, str]) -> tuple[str, str]:
    """(intent, item) without a model, for when the command model is slow or down: the question rules
    below, "I found my ..." as found, and the item by name from the household's saved items."""
    text = strip_wake_phrase(transcript)
    intent = classify(transcript)
    if intent != "ask" and _FOUND.search(text.lower()):
        intent = "found"
    return intent, match_item(text, names)


def classify(transcript: str) -> str:
    """'ask' or 'remember'."""
    text = strip_wake_phrase(transcript)
    low = text.lower().strip()
    if not low:
        return "remember"
    if low.endswith("?") or _QUESTION_START.match(low) or _QUESTION_ANYWHERE.search(low):
        return "ask"
    return "remember"
