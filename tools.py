"""
tools.py — Study tool functions for SmartStudy AI.
Each function calls Groq directly and returns structured data.
"""

import json
import re
from typing import Optional, List, Dict, Any
from groq import Groq
from config import Config


# ── LLM singleton ────────────────────────────────────────────────────────────

_client: Optional[Groq] = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=Config.GROQ_API_KEY)
    return _client


def _llm(prompt: str, system: str = Config.SYSTEM_PROMPT,
         temperature: float = 0.6, max_tokens: int = 2048) -> str:
    """Single Groq LLM call. Returns empty string on error."""
    try:
        resp = _get_client().chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        return f"[AI Error: {exc}]"


# ── JSON extractor ────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Any:
    """
    Try to extract and parse the first valid JSON object or array from text.
    Handles markdown code-fenced JSON (```json … ```) automatically.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { } or [ ]
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return {"raw": text}


# ── 1. Summarize ──────────────────────────────────────────────────────────────

def summarize_text(text: str, subject: str = "", level: str = "intermediate") -> Dict:
    subject_hint = f" for {subject}" if subject else ""
    prompt = f"""Summarize the following study material{subject_hint} for a {level}-level student.

Structure your summary as:
## Overview
(2-3 sentences capturing the core idea)

## Key Concepts
- Concept 1: brief explanation
- Concept 2: brief explanation

## Important Definitions
- **Term**: definition

## Key Takeaways
1. Takeaway one
2. Takeaway two
3. Takeaway three

Study Material:
{text[:6000]}"""
    raw = _llm(prompt)
    return {
        "summary": raw,
        "subject": subject,
        "level": level,
        "word_count": len(text.split()),
        "type": "summary",
    }


# ── 2. Flashcards ─────────────────────────────────────────────────────────────

def generate_flashcards(text: str, count: int = 10, subject: str = "") -> Dict:
    count = min(max(count, 3), 25)
    subject_hint = f" from {subject}" if subject else ""
    prompt = f"""Create exactly {count} high-quality flashcards{subject_hint} from the following material.

You MUST return ONLY a valid JSON object — no explanation, no markdown, no extra text.
Use this exact structure:
{{
  "flashcards": [
    {{
      "id": 1,
      "question": "What is ...?",
      "answer": "...",
      "difficulty": "easy",
      "topic": "..."
    }}
  ]
}}

Difficulty must be one of: easy, medium, hard

Material:
{text[:5000]}"""
    raw = _llm(prompt, temperature=0.5)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict) and "flashcards" in parsed:
        cards = parsed["flashcards"]
        return {"flashcards": cards, "count": len(cards), "subject": subject, "type": "flashcards"}
    return {"flashcards": [], "raw": raw, "subject": subject, "type": "flashcards", "error": "Could not parse flashcards"}


# ── 3. Quiz ───────────────────────────────────────────────────────────────────

def generate_quiz(text: str, count: int = 5, quiz_type: str = "mcq", subject: str = "") -> Dict:
    count = min(max(count, 3), 20)
    subject_hint = f" on {subject}" if subject else ""

    if quiz_type == "mcq":
        fmt = """{
  "questions": [
    {
      "id": 1,
      "question": "Which of the following ...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
      "correct": "A",
      "explanation": "Because ...",
      "topic": "..."
    }
  ]
}"""
    else:
        fmt = """{
  "questions": [
    {"id": 1, "question": "Explain ...", "model_answer": "...", "marks": 5, "topic": "..."}
  ]
}"""

    prompt = f"""Generate exactly {count} {quiz_type.upper()} quiz questions{subject_hint} from the material below.

You MUST return ONLY a valid JSON object — no explanation, no markdown, no extra text.
Use this exact structure:
{fmt}

Material:
{text[:5000]}"""
    raw = _llm(prompt, temperature=0.4)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict) and "questions" in parsed:
        return {
            "questions": parsed["questions"],
            "count": len(parsed["questions"]),
            "quiz_type": quiz_type,
            "subject": subject,
            "type": "quiz",
        }
    return {"questions": [], "raw": raw, "quiz_type": quiz_type, "subject": subject, "type": "quiz", "error": "Could not parse questions"}


# ── 4. Study Plan ─────────────────────────────────────────────────────────────

def create_study_plan(topics: List[str], days_available: int,
                      hours_per_day: float = 2.0,
                      weak_areas: Optional[List[str]] = None) -> Dict:
    weak_areas = weak_areas or []
    topics_str = "\n".join(f"- {t}" for t in topics)
    weak_str   = "\n".join(f"- {w}" for w in weak_areas) if weak_areas else "None identified"

    prompt = f"""Create a detailed day-by-day study plan for a student.

Parameters:
- Days available: {days_available}
- Study hours per day: {hours_per_day}
- Topics:
{topics_str}
- Weak areas:
{weak_str}

You MUST return ONLY a valid JSON object — no explanation, no markdown, no extra text.
{{
  "study_plan": {{
    "total_days": {days_available},
    "hours_per_day": {hours_per_day},
    "daily_schedule": [
      {{
        "day": 1,
        "date_offset": "Day 1",
        "focus_topics": ["Topic A"],
        "tasks": ["Read chapter 1", "Make notes"],
        "hours_allocated": 2.0,
        "review_topics": ["Previous topic"],
        "tip": "Start with the hardest topic when your mind is fresh."
      }}
    ],
    "milestones": [
      {{"day": 3, "milestone": "Completed first half of syllabus"}}
    ],
    "exam_strategy": "..."
  }}
}}"""
    raw = _llm(prompt, temperature=0.4, max_tokens=3000)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict) and "study_plan" in parsed:
        return parsed["study_plan"]
    return {"raw": raw, "topics": topics, "days": days_available}


# ── 5. Predict Exam Topics ────────────────────────────────────────────────────

def predict_exam_topics(text: str, subject: str = "", exam_type: str = "final") -> Dict:
    subject_hint = f" for {subject}" if subject else ""
    prompt = f"""Analyze this study material{subject_hint} and predict the most likely {exam_type} exam topics.

Consider: topic frequency, complexity, foundational importance, and typical exam patterns.

You MUST return ONLY a valid JSON object — no explanation, no markdown, no extra text.
{{
  "predictions": [
    {{
      "topic": "Topic name",
      "confidence": 0.95,
      "urgency": "high",
      "reason": "Appears frequently and is foundational",
      "sub_topics": ["sub1", "sub2"],
      "suggested_prep_time_hours": 3
    }}
  ],
  "overall_advice": "Focus on ...",
  "top_3_focus": ["Topic1", "Topic2", "Topic3"]
}}

urgency must be one of: high, medium, low

Material:
{text[:5000]}"""
    raw = _llm(prompt, temperature=0.3)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict) and "predictions" in parsed:
        return {
            "predictions": parsed["predictions"],
            "advice": parsed.get("overall_advice", ""),
            "top_3": parsed.get("top_3_focus", []),
            "subject": subject,
            "type": "predictions",
        }
    return {"predictions": [], "raw": raw, "subject": subject, "type": "predictions"}


# ── 6. Explain Concept ────────────────────────────────────────────────────────

def explain_concept(concept: str, level: str = "intermediate", subject: str = "") -> Dict:
    subject_hint = f" in {subject}" if subject else ""
    prompt = f"""Explain the concept "{concept}"{subject_hint} for a {level}-level student.

Structure your explanation with these sections:
## Simple Definition
(1-2 sentences)

## Detailed Explanation
(3-4 paragraphs)

## Real-World Examples
- Example 1: ...
- Example 2: ...

## Common Misconceptions
- ...

## Related Concepts
- ...

## Quick Memory Tip
..."""
    raw = _llm(prompt, temperature=0.6, max_tokens=2048)
    return {"explanation": raw, "concept": concept, "level": level, "subject": subject, "type": "explanation"}


# ── 7. Concept Map ────────────────────────────────────────────────────────────

def create_concept_map(text: str, subject: str = "") -> Dict:
    subject_hint = f" for {subject}" if subject else ""
    prompt = f"""Build a hierarchical concept map{subject_hint} from the following material.

You MUST return ONLY a valid JSON object — no explanation, no markdown, no extra text.
{{
  "concept_map": {{
    "root": "Main Topic",
    "nodes": [
      {{
        "id": "node1",
        "label": "Sub-concept",
        "parent": "root",
        "description": "Brief description",
        "children": ["node2"]
      }}
    ],
    "connections": [
      {{"from": "node1", "to": "node2", "relationship": "leads to"}}
    ]
  }}
}}

Material:
{text[:4000]}"""
    raw = _llm(prompt, temperature=0.4)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict) and "concept_map" in parsed:
        return {"concept_map": parsed["concept_map"], "subject": subject, "type": "concept_map"}
    return {"concept_map": {}, "raw": raw, "subject": subject, "type": "concept_map"}


# ── 8. Analyze Weak Areas ─────────────────────────────────────────────────────

def analyze_weak_areas(quiz_results: Dict, subject: str = "") -> Dict:
    if not quiz_results:
        return {"weak_areas": [], "strong_areas": [], "overall_score": 0,
                "study_priority": [], "motivational_message": "Take a quiz first!", "type": "analysis"}

    results_str = "\n".join(f"- {topic}: {score}/100" for topic, score in quiz_results.items())
    prompt = f"""Analyze these quiz performance scores for {subject or 'a student'}:
{results_str}

You MUST return ONLY a valid JSON object — no explanation, no markdown, no extra text.
{{
  "weak_areas": [
    {{
      "topic": "...",
      "score": 45,
      "severity": "critical",
      "recommendations": ["Reread chapter X", "Watch tutorial on Y"],
      "resources": ["Khan Academy", "YouTube"],
      "estimated_fix_time_hours": 3
    }}
  ],
  "strong_areas": ["Topic A", "Topic B"],
  "overall_score": 72,
  "study_priority": ["Topic1", "Topic2"],
  "motivational_message": "You're making great progress!"
}}

severity must be one of: critical, moderate, minor
Only include topics with score < 70 in weak_areas."""
    raw = _llm(prompt, temperature=0.4)
    parsed = _parse_json(raw)
    if isinstance(parsed, dict) and "weak_areas" in parsed:
        return {**parsed, "subject": subject, "type": "analysis"}
    return {"weak_areas": [], "raw": raw, "subject": subject, "type": "analysis"}


# ── File extraction utilities ─────────────────────────────────────────────────

def extract_text_from_pdf(filepath: str) -> str:
    try:
        import PyPDF2
        text = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        result = "\n".join(text).strip()
        return result if result else "No readable text found in PDF."
    except ImportError:
        return "PyPDF2 not installed. Run: pip install PyPDF2"
    except Exception as e:
        return f"Error extracting PDF: {e}"


def extract_text_from_image(filepath: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(filepath)
        result = pytesseract.image_to_string(img).strip()
        return result if result else "No text found in image."
    except ImportError:
        return "pytesseract/Pillow not installed. Run: pip install pytesseract Pillow"
    except Exception as e:
        return f"Error extracting image text: {e}"


def extract_text_from_audio(filepath: str) -> str:
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(filepath) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
    except ImportError:
        return "SpeechRecognition not installed. Run: pip install SpeechRecognition"
    except Exception as e:
        return f"Error transcribing audio: {e}"
