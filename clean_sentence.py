"""
clean_sentence.py

OPTIONAL step. Fingerspelled output looks like "HELLO HOW ARE YOU" -- all
caps, no punctuation. This calls the Anthropic API once on a *finished*
sentence to add casing/punctuation, without changing the actual words.

HOW TO USE:
  1. Set your API key as an environment variable before running anything
     that imports this file:
       export ANTHROPIC_API_KEY=your_key_here   (Mac/Linux)
       setx ANTHROPIC_API_KEY "your_key_here"    (Windows)
  2. Call clean_sentence("HELLO HOW ARE YOU") after a sentence is complete
     in infer.py -- do NOT call this every frame, it's a one-time cleanup
     step per finished sentence, not part of the real-time loop.
"""

import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


def clean_sentence(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                "Add correct casing and punctuation to this fingerspelled "
                f"ASL text. Do not change, add, or remove any words: {raw_text}"
            ),
        }],
    )
    return message.content[0].text
