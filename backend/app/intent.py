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


def classify(transcript: str) -> str:
    """'ask' or 'remember'."""
    text = strip_wake_phrase(transcript)
    low = text.lower().strip()
    if not low:
        return "remember"
    if low.endswith("?") or _QUESTION_START.match(low) or _QUESTION_ANYWHERE.search(low):
        return "ask"
    return "remember"
