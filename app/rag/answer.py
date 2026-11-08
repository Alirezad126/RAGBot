import logging
import re
from pathlib import Path

from app.config import (
    FALLBACK_MODEL,
    GROQ_API_KEY,
    MAX_HISTORY_TURNS,
    MAX_TOKENS,
    MODEL,
    PINNED_PATH,
    REASONING_EFFORT,
)
from app.rag.retrieval import top_chunks

_pinned = None


def pinned():
    """The always-included block, read once per execution environment."""
    global _pinned
    if _pinned is None:
        try:
            _pinned = Path(PINNED_PATH).read_text(encoding="utf-8").strip()
        except OSError:
            logger.warning("No pinned context at %s", PINNED_PATH)
            _pinned = ""
    return _pinned

logger = logging.getLogger(__name__)

CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question, which might reference "
    "earlier turns, rewrite it as a standalone question that can be understood "
    "on its own. Do NOT answer it. Return it unchanged if it already stands "
    "alone. Reply with the question only."
)

# The bot speaks as Alireza on his own portfolio, so the voice matters as much
# as the grounding. Everything it is allowed to claim comes from the excerpts.
ANSWER_PROMPT = """You are Alireza Daneshvar, answering a visitor on your own \
portfolio site. Speak in the first person, plainly and warmly, the way you \
would in a short email.

Answer the question that was asked, specifically. Name the actual project, the \
actual method, the actual number or finding. A reply that lists fields you work \
in or tools you use, without naming what you did, is a failed answer. If you \
are asked what you work on, say what the work is about and what you found, not \
what area it belongs to.

The excerpts below are the only things you know. The block marked "Current, \
always true" is exactly that, and it outranks everything else for any question \
about the present. The rest are labelled with where they come from: your CV, \
your M.A.Sc. thesis, and your profile notes.

Rules about the excerpts, in order of importance:

1. Never state a number, a date, a job title, a venue, or a result that is not \
written in an excerpt. If something is not there, say you have not written that \
down and give your email, alireza.dg1998@gmail.com.
2. The thesis excerpts are one finished project from 2021 to 2023. Never \
present them as current work.
3. A list of tools or skills is not a project. Never assemble a description of \
your work out of one.
4. Every number belongs to the project its own excerpt is about. The 2.7% \
figure is the energy hub thesis, and it means the learned policy came within \
2.7% of the MILP optimum, not that anything was reduced by 2.7%. The 17 microseconds and the 0.1 kB are \
SD-DPC's. The 60.8% is not a result of SD-DPC at all, it is how often the \
distilled baseline hit the obstacle, quoted only to contrast with SD-DPC, which \
hit it never. The 30% and 25% are IranBar. The 10% is the HVAC controller. Never \
move a result from one project to another.

Answer questions about you: your research, your work history, your projects, \
and the everyday things the notes cover, such as where you live or what you \
listen to. Something personal that the notes do answer is fine, so answer it \
briefly and turn back to your work. Only refuse questions that are not about \
you at all, like trivia or current events, and then say that is not what you \
are here for and offer to talk about your work instead.

Answer in three or four sentences, as plain prose. No markdown, no bullet \
points, no equations, and no references to figures or sections, because the \
visitor cannot see the document. End with something that invites a follow-up \
about your work, and never ask how you can assist. Write without semicolons \
and without dashes.

Excerpts:
{context}"""

_client = None


def client():
    global _client
    if _client is None:
        from groq import Groq

        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def complete(messages, max_tokens=MAX_TOKENS, model=MODEL):
    """One completion, retried once on the fallback model.

    Groq retires models from its free catalogue without much notice, so a
    single hardcoded id is a liability. A second, differently sized model
    covers that without putting a picker in front of the visitor.
    """
    try:
        response = client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=max_tokens,
            reasoning_effort=REASONING_EFFORT,
        )
    except Exception:
        if model == FALLBACK_MODEL:
            raise
        logger.warning("Model %s failed, retrying on %s", model, FALLBACK_MODEL)
        response = client().chat.completions.create(
            model=FALLBACK_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=max_tokens,
            reasoning_effort=REASONING_EFFORT,
        )
    return (response.choices[0].message.content or "").strip()


def tidy(text):
    """House style, enforced rather than asked for.

    The prompt tells the model to avoid semicolons and dashes and it mostly
    complies, but "Radiohead; their music" slips through often enough to be
    worth a pass. Em and en dashes become commas, semicolons become full stops,
    and the sentence after one is recapitalised.
    """
    text = re.sub(r"\s*[\u2014\u2013]\s*", ", ", text)
    parts = text.split("; ")
    rest = [part[:1].upper() + part[1:] for part in parts[1:]]
    text = ". ".join([parts[0]] + rest)
    return " ".join(text.replace(";", ".").split())


def as_messages(history):
    """The recent turns, oldest first, as chat messages."""
    recent = history[-MAX_HISTORY_TURNS:]
    return [
        {
            "role": "user" if turn.type == "user" else "assistant",
            "content": turn.message,
        }
        for turn in recent
    ]


def standalone_question(question, history):
    """Fold the conversation into the question so retrieval can stand alone."""
    if not history:
        return question
    messages = (
        [{"role": "system", "content": CONTEXTUALIZE_PROMPT}]
        + as_messages(history)
        + [{"role": "user", "content": question}]
    )
    # A rewrite is one sentence; if the model rambles, fall back to the original.
    rewritten = complete(messages, max_tokens=120)
    return rewritten if rewritten else question


def as_context(chunks):
    retrieved = "\n\n".join(
        f"[{chunk['source']}] {chunk['text']}" for chunk in chunks
    )
    block = pinned()
    if not block:
        return retrieved
    return f"[Current, always true]\n{block}\n\n{retrieved}"


def answer(question, history, chunks, vectors):
    query = standalone_question(question, history)
    context = as_context(top_chunks(query, chunks, vectors))
    messages = (
        [{"role": "system", "content": ANSWER_PROMPT.format(context=context)}]
        + as_messages(history)
        + [{"role": "user", "content": question}]
    )
    return tidy(complete(messages))
