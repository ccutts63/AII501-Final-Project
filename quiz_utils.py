"""Simple, testable helpers for interpreting Gemini quiz output."""

import json


def parse_quiz(response_text: str) -> list[dict]:
    """Accept only three complete four-option MCQs, rejecting malformed output."""
    if not response_text or not response_text.strip():
        raise ValueError("Gemini returned an empty response")
    raw = response_text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    quiz = json.loads(raw)
    if not isinstance(quiz, list) or len(quiz) != 3:
        raise ValueError("Expected exactly 3 quiz questions")

    for index, entry in enumerate(quiz, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Question {index} is not an object")
        if not all(isinstance(entry.get(key), str) and entry[key].strip()
                   for key in ("question", "hint", "explanation")):
            raise ValueError(f"Question {index} is missing required text")
        options = entry.get("options")
        if not isinstance(options, dict) or not all(
            isinstance(options.get(letter), str) and options[letter].strip()
            for letter in "ABCD"
        ):
            raise ValueError(f"Question {index} is missing four options")
        answer = entry.get("answer")
        if not isinstance(answer, str) or answer.upper() not in "ABCD" or len(answer.strip()) != 1:
            raise ValueError(f"Question {index} has an invalid answer key")
        entry["answer"] = answer.upper()
    return quiz
