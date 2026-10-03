"""
FastAPI application providing HTTP endpoints for the RelicLLM project.

Endpoints
---------
GET  /quiz               - Get a welcome message and list of available stocks.
POST /quiz               - Generate a quiz (5 MCQs) for a specific stock.
POST /quiz/submit        - Submit answers and receive a score + eligibility verdict.
POST /assistant          - Chat with the AI financial assistant.
GET  /assistant/history   - Retrieve the stored chat history for a session.
"""

import json
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from llm_core import (
    COMPANY_TICKERS,
    generate_response,
    get_rag_context,
    get_stock_data,
)

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BharatFinanceEdu API",
    description="API for the investment-eligibility quiz and AI financial assistant.",
    version="1.0.0",
)

# Allow frontend apps to call these endpoints
(app.add_middleware)(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# In-memory session stores (replace with Redis/PostgreSQL in production)
# ---------------------------------------------------------------------------

# session_id -> list of {"role": ..., "content": ...}
_quiz_sessions: dict[str, dict] = {}       # quiz_id -> {stock_name, questions, correct_answers}
_assistant_sessions: dict[str, List[dict]] = {}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QuizRequest(BaseModel):
    stock_name: str = Field(..., example="RELIANCE")


class QuizSubmitRequest(BaseModel):
    quiz_id: str = Field(..., example="quiz_001")
    answers: List[str] = Field(..., example=["A", "C", "B", "D", "A"])


class AssistantRequest(BaseModel):
    message: str = Field(..., example="What is a mutual fund?")
    session_id: str = Field("default", example="user_abc123")
    chat_history: Optional[List[dict]] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    reply: str
    sources: List[str] = Field(default_factory=list)
    live_data: bool


class QuizResponse(BaseModel):
    quiz_id: str
    stock_name: str
    questions: List[dict]


class QuizResultResponse(BaseModel):
    score: int
    total: int
    eligible: bool
    correct_answers: List[str]
    feedback: str


# ---------------------------------------------------------------------------
# Helper: extract questions JSON from LLM output
# ---------------------------------------------------------------------------

def _extract_quiz_json(text: str) -> List[dict]:
    """
    The LLM is prompted to return pure JSON.  If it adds any prose we
    try to recover the JSON block.
    """
    # Try direct parse
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array inside ```json ... ```
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Fallback: try finding any JSON array
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse quiz JSON from LLM output:\n{text[:500]}")


# ---------------------------------------------------------------------------
# QUIZ ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/quiz", tags=["Quiz"])
async def get_quiz_info():
    """Return welcome message and list of available stocks for the quiz."""
    return {
        "message": (
            "Welcome to the BharatFinanceEdu Investment Quiz! "
            "Choose a stock, answer 5 multiple-choice questions, "
            "and we'll determine if you're eligible to trade."
        ),
        "available_stocks": {k: v.replace(".NS", "") for k, v in COMPANY_TICKERS.items()},
    }


@app.post("/quiz", response_model=QuizResponse, tags=["Quiz"])
async def generate_quiz(req: QuizRequest):
    """
    Generate 5 multiple-choice questions about *stock_name*.

    The LLM receives:
      - SEBI/RBI regulatory context from ChromaDB
      - Live stock data (price, beta, risk profile)
      - A strict system prompt requiring JSON output with a specific schema.
    """
    stock_name = req.stock_name.strip().upper()
    live_context = get_stock_data(stock_name)
    if not live_context:
        raise HTTPException(
            status_code=404,
            detail=f"Stock '{stock_name}' not found. "
                   f"Available: {list(COMPANY_TICKERS.keys())}"
        )

    # Retrieve relevant RAG chunks
    rag_context = get_rag_context(stock_name + " investment risk", k=3)
    if not rag_context:
        rag_context = get_rag_context("mutual fund investment basics", k=3)

    system_prompt = (
        "You are a financial-literacy quiz generator for Indian investors. "
        "Generate exactly 5 multiple-choice questions about the given stock. "
        "Each question must have exactly 4 options labelled 'A', 'B', 'C', 'D'. "
        "Mark exactly one correct answer per question. "
        "Use the provided live market data and context to make questions realistic. "
        "Output ONLY valid JSON (no prose, no markdown code fences). "
        "JSON schema: "
        '[{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."}, "correct_answer": "A"}, ...]'
    )

    user_prompt = (
        f"Stock: {stock_name}\n"
        f"Live Market Data:\n{live_context}\n"
        f"Regulatory Context:\n{rag_context}"
    )

    raw_output = generate_response(
        system_instruction=system_prompt,
        user_query=user_prompt,
        max_new_tokens=2000,
        temperature=0.2,
    )

    try:
        questions = _extract_quiz_json(raw_output)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Validate questions structure
    if len(questions) != 5:
        raise HTTPException(
            status_code=500,
            detail=f"Expected 5 questions, got {len(questions)}"
        )
    for q in questions:
        if set(q.get("options", {}).keys()) != {"A", "B", "C", "D"}:
            raise HTTPException(
                status_code=500,
                detail=f"Question missing required options: {q.get('question', '???')[:60]}"
            )

    # Generate a unique quiz ID and store correct answers
    import uuid
    quiz_id = f"quiz_{uuid.uuid4().hex[:8]}"
    correct_answers = [q["correct_answer"] for q in questions]

    _quiz_sessions[quiz_id] = {
        "stock_name": stock_name,
        "questions": questions,
        "correct_answers": correct_answers,
    }

    # Return questions WITHOUT correct answers (so frontend can't cheat)
    public_questions = [
        {
            "question": q["question"],
            "options": q["options"],
        }
        for q in questions
    ]

    return QuizResponse(
        quiz_id=quiz_id,
        stock_name=stock_name,
        questions=public_questions,
    )


@app.post("/quiz/submit", response_model=QuizResultResponse, tags=["Quiz"])
async def submit_quiz(req: QuizSubmitRequest):
    """
    Score the user's answers and return eligibility.

    Scoring rule: ≥ 60% → eligible to trade; otherwise not eligible.
    """
    session = _quiz_sessions.get(req.quiz_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Quiz ID '{req.quiz_id}' not found or expired."
        )

    correct_answers = session["correct_answers"]
    user_answers = [a.upper() for a in req.answers]

    if len(user_answers) != len(correct_answers):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(correct_answers)} answers, got {len(user_answers)}."
        )

    score = sum(1 for u, c in zip(user_answers, correct_answers) if u == c)
    total = len(correct_answers)
    percentage = (score / total) * 100
    eligible = percentage >= 60

    # Feedback messages
    feedback_thresholds = [
        (80, "Excellent! You have a strong understanding of financial concepts."),
        (60, "Good job! You understand the basics. Trade responsibly."),
        (40, "Review the material. Consider revisiting financial education."),
        (0, "Beginner level. We recommend studying the content before trading."),
    ]
    _, feedback = next(
        ((t, f) for t, f in feedback_thresholds if percentage >= t),
        (0, "Keep learning!")
    )

    # Clean up the session
    del _quiz_sessions[req.quiz_id]

    return QuizResultResponse(
        score=score,
        total=total,
        eligible=eligible,
        correct_answers=correct_answers,
        feedback=feedback,
    )


# ---------------------------------------------------------------------------
# ASSISTANT ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/assistant", response_model=AssistantResponse, tags=["Assistant"])
async def assistant_chat(req: AssistantRequest):
    """
    Chat with the AI financial assistant.

    Accepts the user message, optional chat history, and a session ID
    for multi-turn conversations.

    The LLM receives:
      - RAG context from ChromaDB (if the question relates to regulation)
      - Live stock data (if the question mentions a tracked company)
    """
    system_instruction = (
        "You are BharatFinanceEdu, a friendly expert AI financial educator for Indian investors. "
        "Use analogies, simple language, and always dispel misconceptions. "
        "Support English, Hindi, and Marathi. "
        "When you use information from official documents, cite the source at the end. "
        "Keep answers concise and actionable."
    )

    rag_context = get_rag_context(req.message, k=3)
    live_context = ""
    live_data = False
    for company in COMPANY_TICKERS:
        if company.lower() in req.message.lower():
            live_context = get_stock_data(req.message)
            live_data = bool(live_context)
            break

    # Build messages list (preserve chat history)
    messages = req.chat_history or []
    messages.append({"role": "user", "content": req.message})

    reply = generate_response(
        system_instruction=system_instruction,
        user_query=req.message,
        context=rag_context,
        live_context=live_context,
        max_new_tokens=500,
        temperature=0.3,
    )

    sources = []
    if rag_context:
        sources.append("SEBI/RBI Knowledge Base (ChromaDB)")

    # Store in memory for this session
    _assistant_sessions[req.session_id] = messages + [
        {"role": "assistant", "content": reply}
    ]

    return AssistantResponse(
        reply=reply,
        sources=sources,
        live_data=live_data,
    )


@app.get("/assistant/history/{session_id}", tags=["Assistant"])
async def get_assistant_history(session_id: str):
    """Retrieve the chat history for a given session ID."""
    if session_id not in _assistant_sessions:
        return {"session_id": session_id, "history": []}
    return {"session_id": session_id, "history": _assistant_sessions[session_id]}


# ---------------------------------------------------------------------------
# Root / health-check
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "BharatFinanceEdu (RelicLLM)",
        "endpoints": {
            "quiz": "/quiz (GET, POST) and /quiz/submit (POST)",
            "assistant": "/assistant (POST) and /assistant/history/{session_id} (GET)",
            "docs": "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
