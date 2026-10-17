"""A household's language: stored and validated, then used for answers, fixed phrases and the summary."""

import asyncio
from urllib.parse import unquote

import pytest
from pydantic import ValidationError

from app import languages
from app.ai import DIDNT_CATCH, FIXED_PHRASES, NO_ANSWER, REFUSED
from app.db_places import Household
from app.routers.households import HouseholdUpdate

from .test_ai_calls import make_ai
from .test_api import CAREGIVER, DEVICE, client, log  # noqa: F401 (client is a fixture)


def ask(client, text):
    return client.post("/api/v1/memory/query", headers=DEVICE, files={"audio": ("q.wav", text.encode(), "audio/wav")})


# ---------------------------------------------------------------- storage and API


def test_language_defaults_to_english():
    assert Household("h1", "Mom", False, "owner").language == "en"
    assert languages.normalize(None) == languages.normalize("xx") == "en"
    assert languages.normalize(" ES ") == "es"


def test_update_accepts_known_codes_and_rejects_unknown():
    assert HouseholdUpdate(language="VI").language == "vi"
    assert HouseholdUpdate(share_public=True).language is None
    for bad in ("klingon", "xx", ""):
        with pytest.raises(ValidationError):
            HouseholdUpdate(language=bad)


@pytest.fixture
def owned(client):  # noqa: F811
    """The fake caregiver owns household h1."""
    db = client.app_state.db
    db.saved = {}

    async def member_role(household_id, user_sub):
        return "owner" if household_id == "h1" else None

    async def set_language(household_id, language):
        db.saved["language"] = language

    async def set_share_public(household_id, share):
        db.saved["share_public"] = share

    db.member_role, db.set_language, db.set_share_public = member_role, set_language, set_share_public
    return client


def test_patch_sets_language(owned):
    resp = owned.patch("/api/v1/households/h1", headers=CAREGIVER, json={"language": "es"})
    assert resp.status_code == 200 and resp.json() == {"language": "es"}
    assert owned.app_state.db.saved == {"language": "es"}


def test_patch_rejects_unknown_language(owned):
    resp = owned.patch("/api/v1/households/h1", headers=CAREGIVER, json={"language": "tlh"})
    assert resp.status_code == 422
    assert owned.app_state.db.saved == {}


def test_patch_share_public_alone_still_works(owned):
    resp = owned.patch("/api/v1/households/h1", headers=CAREGIVER, json={"share_public": True})
    assert resp.json() == {"share_public": True}
    assert owned.app_state.db.saved == {"share_public": True}


# ---------------------------------------------------------------- answers


def test_answer_prompt_names_the_language():
    ai, models = make_ai()
    asyncio.run(ai.answer("¿Dónde están mis llaves?", [], "es"))
    (call,) = models.generated
    assert "Spanish" in call["config"].system_instruction


def test_answer_prompt_defaults_to_english():
    ai, models = make_ai()
    asyncio.run(ai.answer("Where are my keys?", []))
    assert "English" in models.generated[0]["config"].system_instruction


def test_summary_prompt_names_the_language():
    ai, models = make_ai()
    asyncio.run(ai.caregiver_summary("Things put down: 3.", "vi"))
    assert "Vietnamese" in models.generated[0]["config"].system_instruction


@pytest.mark.parametrize("language, answered_by", [(None, "template"), ("en", "template"), ("es", "llm"), ("vi", "llm")])
def test_template_answers_only_in_english(client, language, answered_by):  # noqa: F811
    client.app_state.db.language = language
    log(client, "I am putting my keys on the kitchen counter")
    resp = ask(client, "Where are my keys?")  # confident: English gets the template
    assert resp.headers["x-answered-by"] == answered_by
    if answered_by == "llm":
        assert client.app_state.ai.answered_in == language


def test_spanish_question_finds_a_memory(client):  # noqa: F811
    """The fake vector search returns every memory, standing in for multilingual embeddings:
    a Spanish question reaches a Spanish note even though the keyword search can't help."""
    client.app_state.db.language = "es"
    log(client, "Puse mis llaves en la mesa de la cocina")
    resp = ask(client, "¿Dónde están mis llaves?")
    assert resp.status_code == 200
    question, memories = client.app_state.ai.answered_with
    assert question == "¿Dónde están mis llaves?"
    assert [m.transcript for m in memories] == ["Puse mis llaves en la mesa de la cocina"]


# ---------------------------------------------------------------- fixed phrases and speech


@pytest.mark.parametrize("english", [DIDNT_CATCH, REFUSED, NO_ANSWER])
def test_every_language_has_each_fixed_phrase(english):
    for code in languages.LANGUAGES:
        text = languages.phrase(english, code)
        assert text and (text != english) == (code != "en")
        assert text in FIXED_PHRASES  # so its audio is cached


def test_unknown_language_falls_back_to_english_phrase():
    assert languages.phrase(DIDNT_CATCH, "xx") == DIDNT_CATCH


def test_empty_question_gets_didnt_catch_in_the_household_language(client):  # noqa: F811
    client.app_state.db.language = "vi"
    async def silence(*a):
        return ""

    client.app_state.ai.transcribe = silence
    resp = ask(client, "(no speech)")
    assert unquote(resp.headers["x-answer"]) == languages.phrase(DIDNT_CATCH, "vi")


def test_refusal_is_translated():
    ai, models = make_ai(llm_provider="claude")

    async def refused(*a, **k):
        return None

    ai._generate = refused
    assert asyncio.run(ai.answer("?", [], "fr")) == languages.phrase(REFUSED, "fr")


def test_translated_fixed_phrase_audio_is_cached_and_text_passes_through():
    ai, models = make_ai()
    vi = languages.phrase(DIDNT_CATCH, "vi")
    asyncio.run(ai.speak(vi))
    asyncio.run(ai.speak(vi))
    (call,) = models.generated
    assert call["contents"].endswith(vi)


def test_summary_uses_household_language(client):  # noqa: F811
    client.app_state.db.language = "zh"
    log(client)
    assert client.get("/api/v1/dashboard/summary", headers=CAREGIVER).status_code == 200
    assert client.app_state.ai.summary_language == "zh"
