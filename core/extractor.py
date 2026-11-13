import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
API_DELAY = 1


def get_llm():
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def _call_with_retry(chain, input_data, max_retries=3):
    """Execute chain with retries if throttled."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = (attempt + 1) * 3
                print(f"  ⏳ Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    return chain.invoke(input_data)


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No action items found."

    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
    try:
        return _call_with_retry(chain, transcript[:6000])
    except Exception as e:
        print(f"⚠️ Action items fallback: {e}")
        return "No action items extracted."


def extract_key_decisions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No key decisions found."

    time.sleep(API_DELAY)
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    try:
        return _call_with_retry(chain, transcript[:6000])
    except Exception as e:
        print(f"⚠️ Key decisions fallback: {e}")
        return "No key decisions extracted."


def extract_questions(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "No open questions found."

    time.sleep(API_DELAY)
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    try:
        return _call_with_retry(chain, transcript[:6000])
    except Exception as e:
        print(f"⚠️ Questions extraction fallback: {e}")
        return "No open questions extracted."
