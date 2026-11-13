import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
API_DELAY = 1  # mistral-large-latest supports 50 req/min


def get_llm():
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


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


def generate_title(transcript: str) -> str:
    """Generate a short professional title from transcript."""
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    llm = get_llm()
    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Return only the title text, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    try:
        title = _call_with_retry(title_chain, transcript[:2000])
        return title.strip().strip('"').strip("'")
    except Exception as e:
        print(f"⚠️ Title generation fallback: {e}")
        first_line = transcript.strip().splitlines()[0]
        return " ".join(first_line.split()[:8])


def summarize(transcript: str) -> str:
    """Generate a clean bullet-point summary of the transcript."""
    if not transcript or not transcript.strip():
        return "No transcript content available to summarize."

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a transcript concisely in 3-5 bullet points."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # If single chunk, summarize directly
    if len(chunks) == 1:
        try:
            return _call_with_retry(map_chain, {"text": chunks[0]})
        except Exception as e:
            print(f"⚠️ Summarization fallback: {e}")
            return transcript[:500] + "..."

    # Multi-chunk map-reduce
    print(f"  Summarizing {len(chunks)} chunks...")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  → Chunk {i + 1}/{len(chunks)}...")
        try:
            summary = _call_with_retry(map_chain, {"text": chunk})
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"  ⚠️ Chunk {i + 1} summary error: {e}")
            chunk_summaries.append(chunk[:200])

        if i < len(chunks) - 1:
            time.sleep(API_DELAY)

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    try:
        return _call_with_retry(combined_chain, combined)
    except Exception as e:
        print(f"⚠️ Combine summary fallback: {e}")
        return combined
