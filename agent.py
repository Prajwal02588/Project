"""
agent.py — Agentic AI core for SmartStudy Generator.
Uses Groq API with openai/gpt-oss-120b.
Implements a ReAct-style loop: LLM picks a tool → executes → observes → repeats.
"""

import json
import traceback
from typing import Optional, List, Dict, Any
from groq import Groq
from config import Config
from tools import (
    summarize_text,
    generate_flashcards,
    generate_quiz,
    create_study_plan,
    predict_exam_topics,
    explain_concept,
    create_concept_map,
    analyze_weak_areas,
)

# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize study text into structured bullet points with key concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":    {"type": "string", "description": "The text to summarize."},
                    "subject": {"type": "string", "description": "Subject area e.g. Physics."},
                    "level":   {"type": "string", "description": "beginner / intermediate / advanced"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_flashcards",
            "description": "Create Q&A flashcards from study material for active recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":    {"type": "string", "description": "Source material."},
                    "count":   {"type": "integer", "description": "Number of flashcards (default 10)."},
                    "subject": {"type": "string", "description": "Subject area."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz",
            "description": "Generate MCQ or short-answer quiz questions from material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":      {"type": "string", "description": "Source material."},
                    "count":     {"type": "integer", "description": "Number of questions (default 5)."},
                    "quiz_type": {"type": "string", "description": "mcq or short_answer"},
                    "subject":   {"type": "string", "description": "Subject area."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_study_plan",
            "description": "Build a personalized day-by-day study plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topics":         {"type": "array", "items": {"type": "string"}, "description": "Topics to study."},
                    "days_available": {"type": "integer", "description": "Days until exam."},
                    "hours_per_day":  {"type": "number",  "description": "Study hours per day."},
                    "weak_areas":     {"type": "array", "items": {"type": "string"}, "description": "Weak topics."},
                },
                "required": ["topics", "days_available"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_exam_topics",
            "description": "Predict the most likely exam topics with confidence scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":      {"type": "string", "description": "Syllabus or study content."},
                    "subject":   {"type": "string", "description": "Subject area."},
                    "exam_type": {"type": "string", "description": "midterm / final / board"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_concept",
            "description": "Explain a concept clearly with examples and analogies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "The concept to explain."},
                    "level":   {"type": "string", "description": "beginner / intermediate / advanced"},
                    "subject": {"type": "string", "description": "Subject area."},
                },
                "required": ["concept"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_concept_map",
            "description": "Generate a hierarchical concept map from study material.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":    {"type": "string", "description": "Source material."},
                    "subject": {"type": "string", "description": "Subject area."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_weak_areas",
            "description": "Analyze quiz scores to identify weak areas and give recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quiz_results": {"type": "object", "description": "topic -> score (0-100) mapping."},
                    "subject":      {"type": "string", "description": "Subject area."},
                },
                "required": ["quiz_results"],
            },
        },
    },
]

TOOL_FUNCTIONS: Dict[str, Any] = {
    "summarize_text":     summarize_text,
    "generate_flashcards": generate_flashcards,
    "generate_quiz":      generate_quiz,
    "create_study_plan":  create_study_plan,
    "predict_exam_topics": predict_exam_topics,
    "explain_concept":    explain_concept,
    "create_concept_map": create_concept_map,
    "analyze_weak_areas": analyze_weak_areas,
}


# ── SmartStudyAgent ────────────────────────────────────────────────────────────

class SmartStudyAgent:
    """
    Agentic AI backed by Groq's openai/gpt-oss-120b.
    Uses a ReAct loop: think → pick tool → execute → observe → repeat → respond.
    Falls back to direct LLM answer if tool calling isn't triggered.
    """

    def __init__(self):
        if not Config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Create a .env file with GROQ_API_KEY=your_key_here "
                "and get a free key at https://console.groq.com"
            )
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        self.model  = Config.MODEL_NAME
        self.conversation_history: List[Dict] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str, context: Optional[Dict] = None) -> Dict:
        """General-purpose agentic chat. Returns response dict."""
        try:
            messages = self._build_messages(user_message, context)
            return self._run_agent_loop(messages)
        except Exception as exc:
            tb = traceback.format_exc()
            return {
                "response": f"I encountered an error: {exc}\n\nPlease check your GROQ_API_KEY in the .env file.",
                "tool_calls": [],
                "type": "error",
                "detail": tb,
            }

    def process_document(self, text: str, action: str,
                         options: Optional[Dict] = None) -> Dict:
        """Process text with a specific action."""
        options = options or {}
        try:
            handlers = {
                "summarize":      self._action_summarize,
                "flashcards":     self._action_flashcards,
                "quiz":           self._action_quiz,
                "concept_map":    self._action_concept_map,
                "predict_topics": self._action_predict_topics,
            }
            handler = handlers.get(action, self._action_summarize)
            return handler(text, options)
        except Exception as exc:
            return {"response": f"Error processing document: {exc}", "type": "error"}

    def generate_study_plan(self, topics: List[str], days: int,
                            hours: float, weak_areas: Optional[List[str]] = None) -> Dict:
        try:
            result = create_study_plan(topics, days, hours, weak_areas or [])
            return self._llm_enhance(
                result,
                "Format this study plan beautifully with clear day-by-day breakdown, milestones, and exam tips."
            )
        except Exception as exc:
            return {"response": f"Error generating study plan: {exc}", "type": "error"}

    def get_exam_predictions(self, content: str, subject: str,
                             exam_type: str = "final") -> Dict:
        try:
            result = predict_exam_topics(content, subject, exam_type)
            return self._llm_enhance(
                result,
                "Present these predictions as a prioritized study alert list with 🔴 HIGH, 🟡 MEDIUM, 🟢 LOW urgency."
            )
        except Exception as exc:
            return {"response": f"Error generating predictions: {exc}", "type": "error"}

    def analyze_performance(self, quiz_results: Dict, subject: str) -> Dict:
        try:
            result = analyze_weak_areas(quiz_results, subject)
            return self._llm_enhance(
                result,
                "Give personalized, encouraging improvement recommendations based on these weak areas."
            )
        except Exception as exc:
            return {"response": f"Error analyzing performance: {exc}", "type": "error"}

    def reset_conversation(self):
        self.conversation_history = []

    # ── Agent loop ─────────────────────────────────────────────────────────────

    def _run_agent_loop(self, messages: List[Dict],
                        max_iterations: int = 5) -> Dict:
        tool_calls_log: List[Dict] = []

        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                )
            except Exception as exc:
                # If tool_choice fails (model doesn't support it), retry without tools
                if "tool" in str(exc).lower() or "function" in str(exc).lower():
                    return self._run_without_tools(messages)
                raise

            message = response.choices[0].message

            # ── No tool call → final answer ────────────────────────────────
            if not message.tool_calls:
                content = message.content or ""
                self.conversation_history.append({"role": "assistant", "content": content})
                return {
                    "response": content,
                    "tool_calls": tool_calls_log,
                    "type": "chat",
                }

            # ── Append assistant turn with tool_calls ──────────────────────
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            # ── Execute each tool ──────────────────────────────────────────
            for tc in message.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}

                fn = TOOL_FUNCTIONS.get(fn_name)
                if fn:
                    try:
                        tool_result = fn(**args)
                    except Exception as e:
                        tool_result = {"error": str(e), "tool": fn_name}
                else:
                    tool_result = {"error": f"Unknown tool: {fn_name}"}

                tool_calls_log.append({"tool": fn_name, "args": args, "result": tool_result})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, default=str),
                })

        # Exhausted iterations — ask for final answer explicitly
        messages.append({
            "role": "user",
            "content": "Please provide your final answer based on the information gathered.",
        })
        try:
            final = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=Config.MAX_TOKENS,
                temperature=0.5,
            )
            content = final.choices[0].message.content or "Processing complete."
        except Exception:
            content = "I processed your request across multiple steps."

        self.conversation_history.append({"role": "assistant", "content": content})
        return {"response": content, "tool_calls": tool_calls_log, "type": "chat"}

    def _run_without_tools(self, messages: List[Dict]) -> Dict:
        """Fallback: direct LLM call without function calling."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
            )
            content = response.choices[0].message.content or ""
            self.conversation_history.append({"role": "assistant", "content": content})
            return {"response": content, "tool_calls": [], "type": "chat"}
        except Exception as exc:
            return {"response": f"AI Error: {exc}", "tool_calls": [], "type": "error"}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_messages(self, user_message: str,
                        context: Optional[Dict]) -> List[Dict]:
        messages: List[Dict] = [{"role": "system", "content": Config.SYSTEM_PROMPT}]
        if context:
            ctx_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            messages.append({"role": "system", "content": f"Student Context:\n{ctx_str}"})
        # Include last 10 turns to avoid token overflow
        messages.extend(self.conversation_history[-10:])
        messages.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "user", "content": user_message})
        return messages

    def _llm_enhance(self, raw_result: Dict, instruction: str) -> Dict:
        """Ask the LLM to format a tool result into a nicely structured response."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": Config.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"{instruction}\n\nData:\n{json.dumps(raw_result, indent=2, default=str)}",
                    },
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=0.5,
            )
            content = resp.choices[0].message.content or json.dumps(raw_result)
        except Exception as exc:
            content = json.dumps(raw_result, indent=2, default=str)
        return {"response": content, "data": raw_result, "type": "structured"}

    # ── Action handlers (called by process_document) ──────────────────────────

    def _action_summarize(self, text: str, options: Dict) -> Dict:
        result = summarize_text(
            text,
            options.get("subject", ""),
            options.get("level", "intermediate"),
        )
        return self._llm_enhance(
            result,
            "Present this summary beautifully with clear sections, key concepts, and takeaways."
        )

    def _action_flashcards(self, text: str, options: Dict) -> Dict:
        result = generate_flashcards(
            text,
            options.get("count", 10),
            options.get("subject", ""),
        )
        # Also get a formatted narrative
        enhanced = self._llm_enhance(result, "Briefly introduce these flashcards and note the difficulty distribution.")
        enhanced["data"] = result   # Keep raw data for the flashcard viewer
        return enhanced

    def _action_quiz(self, text: str, options: Dict) -> Dict:
        result = generate_quiz(
            text,
            options.get("count", 5),
            options.get("quiz_type", "mcq"),
            options.get("subject", ""),
        )
        return {"response": "", "data": result, "type": "quiz"}

    def _action_concept_map(self, text: str, options: Dict) -> Dict:
        result = create_concept_map(text, options.get("subject", ""))
        return {"response": "", "data": result, "type": "concept_map"}

    def _action_predict_topics(self, text: str, options: Dict) -> Dict:
        result = predict_exam_topics(
            text,
            options.get("subject", ""),
            options.get("exam_type", "final"),
        )
        return self._llm_enhance(
            result,
            "Present these exam predictions as prioritized study alerts. Use 🔴 HIGH, 🟡 MEDIUM, 🟢 LOW urgency labels."
        )
