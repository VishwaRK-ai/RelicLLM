"""BharatFinanceEdu impulsivity gate demo.

Run the UI with ``streamlit run trade_check.py`` or verify the deterministic
pipeline with ``python trade_check.py --self-test``. Build cached bank audio
ahead of a demo with ``python trade_check.py --build-audio-cache``.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


# Gate bands are deliberately transparent and easy to tune for the demo.
WARN_THRESHOLD = 35
DELAY_THRESHOLD = 55
BLOCK_THRESHOLD = 75
QUIZ_PASS_RATIO = 2 / 3
COOLDOWN_SECONDS = {"delay": 30, "block": 60}
AUDIO_CACHE_DIR = Path.home() / ".cache" / "bharat_finance_edu" / "trade_check_audio"

@dataclass(frozen=True)
class TradeOrder:
    symbol: str
    amount: float
    portfolio_value: float
    recent_trades: int
    price_change_pct: float
    urgency: bool = False
    social_tip: bool = False

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("A stock symbol is required.")
        if not math.isfinite(self.amount) or self.amount <= 0:
            raise ValueError("Trade amount must be a positive number.")
        if not math.isfinite(self.portfolio_value) or self.portfolio_value <= 0:
            raise ValueError("Portfolio value must be a positive number.")
        if self.amount > self.portfolio_value:
            raise ValueError("Trade amount cannot exceed the portfolio value.")
        if self.recent_trades < 0:
            raise ValueError("Recent trade count cannot be negative.")
        if not math.isfinite(self.price_change_pct):
            raise ValueError("Price change must be a finite number.")


@dataclass(frozen=True)
class QuizQuestion:
    question_id: str
    stem: str
    options: tuple[str, ...]
    answer_index: int
    explanation: str

    def __post_init__(self) -> None:
        if not self.stem.strip() or not 2 <= len(self.options) <= 4:
            raise ValueError("Quiz questions need a stem and two to four options.")
        if not 0 <= self.answer_index < len(self.options):
            raise ValueError("Question answer index must identify an option.")


QUESTION_BANK: dict[str, tuple[QuizQuestion, ...]] = {
    "en": (
        QuizQuestion(
            "diversification",
            "Why spread investments across different assets?",
            ("To reduce concentration risk", "To guarantee a profit", "To avoid all losses", "To copy a popular tip"),
            0,
            "Diversification can reduce the impact of one investment performing poorly; it cannot guarantee returns.",
        ),
        QuizQuestion(
            "hype",
            "A stock is trending online. What is a careful next step?",
            ("Buy before it rises", "Check reliable information and your plan", "Borrow to buy more", "Ignore every risk"),
            1,
            "A popular post is not proof of value. Check reliable information and whether the trade fits your plan.",
        ),
        QuizQuestion(
            "loss",
            "A stock falls sharply. What should guide your next move?",
            ("A rushed reaction", "A guaranteed tip", "Your goals and risk plan", "A stranger's prediction"),
            2,
            "A decision based on goals and a risk plan is less reactive than acting on a prediction or a sudden price move.",
        ),
        QuizQuestion(
            "cooling_off",
            "You feel pressure to trade right now. What can help?",
            ("Pause and review the trade", "Increase the order", "Skip your research", "Follow the crowd"),
            0,
            "Taking a pause gives you time to review the risks and avoid a decision made under pressure.",
        ),
    ),
    "hi": (
        QuizQuestion(
            "diversification",
            "निवेश को अलग-अलग जगह बाँटना क्यों उपयोगी है?",
            ("एक जगह का जोखिम घटाने के लिए", "मुनाफ़े की गारंटी के लिए", "हर नुकसान रोकने के लिए", "लोकप्रिय सलाह मानने के लिए"),
            0,
            "विविध निवेश एक निवेश के खराब प्रदर्शन का असर घटा सकते हैं, लेकिन मुनाफ़े की गारंटी नहीं देते।",
        ),
        QuizQuestion(
            "hype",
            "कोई शेयर ऑनलाइन लोकप्रिय है। समझदारी भरा अगला कदम क्या है?",
            ("कीमत बढ़ने से पहले खरीदें", "भरोसेमंद जानकारी और योजना जाँचें", "उधार लेकर खरीदें", "सभी जोखिम भूल जाएँ"),
            1,
            "लोकप्रिय पोस्ट मूल्य का प्रमाण नहीं है। भरोसेमंद जानकारी और अपनी योजना जाँचें।",
        ),
        QuizQuestion(
            "loss",
            "शेयर की कीमत तेज़ी से गिरे, तो अगला कदम किस आधार पर लें?",
            ("जल्दबाज़ी में", "पक्की सलाह से", "अपने लक्ष्य और जोखिम योजना से", "किसी अनजान की भविष्यवाणी से"),
            2,
            "लक्ष्य और जोखिम योजना के आधार पर निर्णय लेना अचानक कीमत या भविष्यवाणी पर प्रतिक्रिया देने से बेहतर है।",
        ),
        QuizQuestion(
            "cooling_off",
            "अभी तुरंत ट्रेड करने का दबाव लगे, तो क्या मदद कर सकता है?",
            ("रुककर ट्रेड की समीक्षा करना", "ऑर्डर बढ़ाना", "जाँच छोड़ देना", "भीड़ का अनुसरण करना"),
            0,
            "थोड़ा रुकने से जोखिम की समीक्षा करने और दबाव में फैसला लेने से बचने का समय मिलता है।",
        ),
    ),
}
TTS_LANGUAGES = {"en": "English", "hi": "Hindi"}


def generate_synthetic_trade(persona: str) -> TradeOrder:
    """Return a reproducible demo order for one of the named personas."""
    personas = {
        "planned": TradeOrder("DEMO", 5_000, 250_000, 1, 0.5),
        "momentum_chaser": TradeOrder(
            "DEMO", 20_000, 100_000, 4, 6.0, urgency=True, social_tip=True
        ),
        "all_in_fomo": TradeOrder(
            "DEMO", 75_000, 100_000, 8, 12.0, urgency=True, social_tip=True
        ),
    }
    try:
        return personas[persona]
    except KeyError as exc:
        raise ValueError(f"Unknown synthetic persona: {persona}") from exc


def extract_features(order: TradeOrder) -> dict[str, float]:
    """Extract transparent, bounded signals from an order and recent activity."""
    exposure_pct = 100 * order.amount / order.portfolio_value
    return {
        "concentration": min(exposure_pct / 50, 1) * 35,
        "trade_frequency": min(order.recent_trades / 8, 1) * 25,
        "positive_momentum": min(max(order.price_change_pct, 0) / 10, 1) * 20,
        "urgency_language": 15.0 if order.urgency else 0.0,
        "social_tip": 5.0 if order.social_tip else 0.0,
    }


def score_impulsivity(features: Mapping[str, float]) -> int:
    """Combine feature points into an integer score from zero to one hundred."""
    required = {
        "concentration",
        "trade_frequency",
        "positive_momentum",
        "urgency_language",
        "social_tip",
    }
    if set(features) != required:
        raise ValueError("Impulsivity features do not match the supported feature set.")
    if any(not math.isfinite(value) or value < 0 for value in features.values()):
        raise ValueError("Impulsivity features must be finite and non-negative.")
    return round(min(sum(features.values()), 100))


def classify_risk(score: int) -> str:
    if not 0 <= score <= 100:
        raise ValueError("Impulsivity score must be between zero and one hundred.")
    if score >= BLOCK_THRESHOLD:
        return "block"
    if score >= DELAY_THRESHOLD:
        return "delay"
    if score >= WARN_THRESHOLD:
        return "warn"
    return "allow"


def get_quiz_questions(language: str = "en", count: int = 3) -> list[QuizQuestion]:
    if language not in QUESTION_BANK:
        raise ValueError(f"Unsupported quiz language: {language}")
    if count < 1 or count > len(QUESTION_BANK[language]):
        raise ValueError("Question count must fit the fixed question bank.")
    return list(QUESTION_BANK[language][:count])


def assemble_quiz(
    language: str = "en",
    count: int = 3,
    contextual_order: TradeOrder | None = None,
) -> list[QuizQuestion]:
    questions = get_quiz_questions(language, count)
    if contextual_order is not None:
        questions.append(generate_contextual_question(language, contextual_order))
    return questions


def generate_contextual_question(language: str, order: TradeOrder | None = None) -> QuizQuestion:
    """Ask the LLM for a stem only; fixed options and rubric keep scoring deterministic."""
    if language not in QUESTION_BANK:
        raise ValueError(f"Unsupported quiz language: {language}")
    symbol = order.symbol if order else "this stock"
    exposure = (
        round(100 * order.amount / order.portfolio_value)
        if order
        else None
    )
    detail = f"The proposed order is {exposure}% of the portfolio." if exposure is not None else ""
    language_instruction = (
        "Write the question in Hindi using Devanagari script."
        if language == "hi"
        else "Write the question in English."
    )
    prompt = (
        "Write exactly one short, speakable question stem (at most twenty words) "
        "that asks why a person should review concentration and diversification "
        f"before buying {symbol}. {detail} {language_instruction} "
        "Do not write options, numbering, or an answer."
    )
    from llm_core import generate_response

    response = generate_response(
        "You write concise financial-literacy quiz question stems only.",
        prompt,
        max_new_tokens=60,
        temperature=0.2,
    )
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines and re.fullmatch(r"(?:question|stem)\s*:?", lines[0], flags=re.IGNORECASE):
        lines = lines[1:]
    if not lines:
        raise ValueError("The model did not return a usable contextual question stem.")
    stem = re.sub(r"^(?:[-*]\s*|\d+[.)]\s*)", "", lines[0])
    stem = re.sub(r"^(?:question|stem)\s*:\s*", "", stem, flags=re.IGNORECASE)
    stem = stem.strip().strip("\"' ")
    if len(stem.split()) > 20:
        raise ValueError("The model returned a contextual question stem that is too long.")
    stem = stem.rstrip(" .!?") + "?"
    if not stem or stem == "?":
        raise ValueError("The model did not return a usable contextual question stem.")
    if language == "hi":
        options = ("एक जगह का जोखिम घटाने के लिए", "मुनाफ़े की गारंटी के लिए", "हर नुकसान रोकने के लिए", "लोकप्रिय सलाह मानने के लिए")
        answer = 0
        explanation = "अलग-अलग निवेश जोखिम को बाँट सकते हैं, लेकिन मुनाफ़े की गारंटी नहीं देते।"
    else:
        options = ("To reduce concentration risk", "To guarantee a profit", "To avoid all losses", "To copy a popular tip")
        answer = 0
        explanation = "Diversification may reduce concentration risk, but it cannot guarantee a profit."
    return QuizQuestion("contextual_diversification", stem, options, answer, explanation)


def score_quiz(questions: Sequence[QuizQuestion], answers: Sequence[int | None]) -> dict[str, Any]:
    if len(questions) != len(answers) or not questions:
        raise ValueError("Provide one answer for each quiz question.")
    correct = sum(answer == question.answer_index for question, answer in zip(questions, answers))
    required_correct = math.ceil(len(questions) * QUIZ_PASS_RATIO)
    return {
        "correct": correct,
        "total": len(questions),
        "required_correct": required_correct,
        "passed": correct >= required_correct,
    }


def decide_outcome(risk_level: str, quiz_passed: bool | None = None) -> str:
    """Apply the fixed risk/quiz matrix; a critical score cannot be quiz-cleared."""
    if risk_level == "allow":
        return "allow"
    if risk_level not in {"warn", "delay", "block"}:
        raise ValueError(f"Unknown risk level: {risk_level}")
    if quiz_passed is None:
        raise ValueError("A triggered quiz must be completed before deciding.")
    if risk_level == "block":
        return "block"
    if risk_level == "delay":
        return "delay" if quiz_passed else "block"
    return "warn" if quiz_passed else "delay"


def cooldown_remaining(unlock_timestamp: float | None, now: float | None = None) -> int:
    if unlock_timestamp is None:
        return 0
    return max(0, math.ceil(unlock_timestamp - (time.time() if now is None else now)))


def explanation_text(score: int, outcome: str, features: Mapping[str, float]) -> str:
    notable = sorted(features.items(), key=lambda item: item[1], reverse=True)
    signals = [name.replace("_", " ") for name, value in notable if value > 0][:2]
    signal_text = ", ".join(signals) if signals else "few impulsivity signals"
    messages = {
        "allow": "No quiz was needed.",
        "warn": "Review your plan before placing the order.",
        "delay": "Pause before placing another order and review the risks.",
        "block": "This order is blocked for the cooling-off period.",
    }
    return (
        f"The impulsivity score is {score} out of one hundred, with {signal_text}. "
        f"{messages[outcome]} This educational demo is not investment advice."
    )


def synthesize_audio(text: str, language: str, cache_key: str) -> Path:
    if language not in TTS_LANGUAGES:
        raise ValueError(f"Unsupported speech language: {language}")
    AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{cache_key}\0{text}".encode("utf-8")).hexdigest()[:16]
    audio_path = AUDIO_CACHE_DIR / f"{language}_{digest}.mp3"
    if audio_path.exists():
        return audio_path
    from gtts import gTTS

    temporary_path = audio_path.with_suffix(".tmp.mp3")
    gTTS(text=text, lang=language, slow=False).save(str(temporary_path))
    temporary_path.replace(audio_path)
    return audio_path


def question_audio_text(question: QuizQuestion) -> str:
    return f"{question.stem} " + " ".join(
        f"Option {index + 1}. {option}." for index, option in enumerate(question.options)
    )


def question_audio_cache_key(question: QuizQuestion, language: str) -> str:
    if question.question_id == "contextual_diversification":
        return f"contextual:{question.question_id}:{language}:{question.stem}"
    return f"bank:{question.question_id}:{language}"


def build_audio_cache(languages: Sequence[str] = ("en", "hi")) -> int:
    created_or_cached = 0
    for language in languages:
        if language not in QUESTION_BANK:
            raise ValueError(f"Unsupported quiz language: {language}")
        for question in QUESTION_BANK[language]:
            synthesize_audio(
                question_audio_text(question),
                language,
                f"bank:{question.question_id}:{language}",
            )
            created_or_cached += 1
    return created_or_cached


def _run_self_test() -> None:
    planned = score_impulsivity(extract_features(generate_synthetic_trade("planned")))
    chaser = score_impulsivity(extract_features(generate_synthetic_trade("momentum_chaser")))
    all_in = score_impulsivity(extract_features(generate_synthetic_trade("all_in_fomo")))
    assert planned < WARN_THRESHOLD, planned
    assert WARN_THRESHOLD <= chaser < BLOCK_THRESHOLD, chaser
    assert all_in >= BLOCK_THRESHOLD, all_in
    assert classify_risk(planned) == "allow"
    assert decide_outcome("warn", True) == "warn"
    assert decide_outcome("warn", False) == "delay"
    assert decide_outcome("delay", True) == "delay"
    assert decide_outcome("delay", False) == "block"
    assert decide_outcome("block", True) == "block"
    questions = get_quiz_questions()
    result = score_quiz(questions, [question.answer_index for question in questions])
    assert result["passed"] and result["correct"] == 3
    assert len(assemble_quiz()) == 3
    assert cooldown_remaining(130, now=100) == 30
    print(f"Self-test passed: planned={planned}, momentum_chaser={chaser}, all_in_fomo={all_in}.")


def _run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="BharatFinanceEdu Trade Check", page_icon="📈")
    st.title("BharatFinanceEdu: Impulsivity Gate")
    st.caption(
        "A synthetic, educational trade check. The score and gate are deterministic; "
        "the optional contextual question and explanation use the existing LLM."
    )
    state = st.session_state
    if "trade_flow" not in state:
        state.trade_flow = {"phase": "order"}
    if "cooldown_until" not in state:
        state.cooldown_until = None
    flow = state.trade_flow

    def start_order(order: TradeOrder, language: str, contextual: bool) -> None:
        remaining = cooldown_remaining(state.cooldown_until)
        if remaining:
            flow.clear()
            flow.update(phase="cooldown", remaining=remaining)
            return
        features = extract_features(order)
        score = score_impulsivity(features)
        risk_level = classify_risk(score)
        flow.clear()
        flow.update(
            phase="gate",
            order=order,
            features=features,
            score=score,
            risk_level=risk_level,
            language=language,
            contextual=contextual,
        )

    if flow["phase"] == "order":
        st.subheader("Enter an order")
        with st.form("trade_order"):
            persona = st.selectbox(
                "Synthetic demo persona",
                ("Custom order", "Planned investor", "Momentum chaser", "All-in FOMO"),
            )
            symbol = st.text_input("Stock symbol", value="DEMO")
            col1, col2 = st.columns(2)
            amount = col1.number_input("Order amount (₹)", min_value=1.0, value=20_000.0, step=1_000.0)
            portfolio = col2.number_input("Portfolio value (₹)", min_value=1.0, value=100_000.0, step=5_000.0)
            recent_trades = st.number_input("Trades in the last day", min_value=0, max_value=100, value=4)
            price_change = st.number_input("Recent price change (percent)", value=6.0, step=0.5)
            col3, col4 = st.columns(2)
            urgency = col3.checkbox("Feeling urgency to buy")
            social_tip = col4.checkbox("Trade prompted by a social-media tip")
            language = st.selectbox(
                "Quiz and speech language",
                tuple(TTS_LANGUAGES),
                format_func=lambda code: TTS_LANGUAGES[code],
            )
            contextual = st.checkbox("Add one LLM-generated contextual question")
            submitted = st.form_submit_button("Check trade")
        if submitted:
            try:
                if persona != "Custom order":
                    key = {
                        "Planned investor": "planned",
                        "Momentum chaser": "momentum_chaser",
                        "All-in FOMO": "all_in_fomo",
                    }[persona]
                    order = generate_synthetic_trade(key)
                else:
                    order = TradeOrder(
                        symbol=symbol,
                        amount=amount,
                        portfolio_value=portfolio,
                        recent_trades=int(recent_trades),
                        price_change_pct=price_change,
                        urgency=urgency,
                        social_tip=social_tip,
                    )
                start_order(order, language, contextual)
                st.rerun()
            except ValueError as error:
                st.error(str(error))

    elif flow["phase"] == "gate":
        st.subheader("Gate check")
        st.metric("Impulsivity score", f"{flow['score']} / 100")
        st.write(f"**Risk band:** {flow['risk_level'].upper()}")
        with st.expander("How the score is calculated"):
            st.write(
                "Concentration (up to 35), recent trade frequency (up to 25), "
                "positive price momentum (up to 20), urgency (15), and a social tip (5)."
            )
            st.json(flow["features"])
        if flow["risk_level"] == "allow":
            st.success("The score is below the quiz threshold.")
            if st.button("Continue to outcome"):
                flow["outcome"] = "allow"
                flow["phase"] = "decision"
                st.rerun()
        else:
            st.info("A short knowledge check is required before the gate decision.")
            if st.button("Begin knowledge check"):
                questions = assemble_quiz(
                    flow["language"],
                    contextual_order=flow["order"] if flow["contextual"] else None,
                )
                flow["questions"] = questions
                flow["answers"] = [None] * len(questions)
                flow["question_index"] = 0
                flow["phase"] = "quiz_listen"
                st.rerun()

    elif flow["phase"] == "cooldown":
        remaining = cooldown_remaining(state.cooldown_until)
        st.warning(f"Cooling-off period is active. Try again in about {remaining} seconds.")
        if remaining == 0:
            state.cooldown_until = None
            flow.clear()
            flow["phase"] = "order"
            st.rerun()
        if st.button("Return to order entry"):
            flow.clear()
            flow["phase"] = "order"
            st.rerun()

    elif flow["phase"] in {"quiz_listen", "quiz_answer"}:
        questions = flow["questions"]
        index = flow["question_index"]
        question = questions[index]
        st.subheader(f"Knowledge check {index + 1} of {len(questions)}")
        st.write(question.stem)
        speech_text = question_audio_text(question)
        audio_path = synthesize_audio(
            speech_text,
            flow["language"],
            question_audio_cache_key(question, flow["language"]),
        )
        st.audio(str(audio_path), format="audio/mp3")
        if flow["phase"] == "quiz_listen":
            if st.button("Audio finished — start answer timer"):
                flow["question_started_at"] = time.time()
                flow["phase"] = "quiz_answer"
                st.rerun()
        else:
            selected = st.radio(
                "Choose one answer",
                range(len(question.options)),
                format_func=lambda option_index: question.options[option_index],
                key=f"answer_{index}",
            )
            if st.button("Submit answer"):
                elapsed = max(0.0, time.time() - flow["question_started_at"])
                flow["answers"][index] = selected
                flow.setdefault("response_seconds", []).append(round(elapsed, 2))
                if index + 1 < len(questions):
                    flow["question_index"] += 1
                    flow["phase"] = "quiz_listen"
                else:
                    result = score_quiz(questions, flow["answers"])
                    flow["quiz_result"] = result
                    flow["outcome"] = decide_outcome(flow["risk_level"], result["passed"])
                    flow["phase"] = "decision"
                st.rerun()

    elif flow["phase"] == "decision":
        outcome = flow["outcome"]
        if outcome in COOLDOWN_SECONDS:
            state.cooldown_until = time.time() + COOLDOWN_SECONDS[outcome]
        flow["phase"] = "outcome"
        st.rerun()

    elif flow["phase"] == "outcome":
        outcome = flow["outcome"]
        st.subheader("Outcome")
        st.metric("Impulsivity score", f"{flow['score']} / 100")
        st.write(f"**Gate decision:** {outcome.upper()}")
        st.write(explanation_text(flow["score"], outcome, flow["features"]))
        if "quiz_result" in flow:
            result = flow["quiz_result"]
            st.write(
                f"Quiz: {result['correct']} of {result['total']} correct "
                f"(at least {result['required_correct']} needed to pass)."
            )
            st.caption(
                "Answer time after audio: "
                f"{', '.join(map(str, flow.get('response_seconds', [])))} seconds."
            )
            with st.expander("Quiz answers and rubric"):
                for question, answer in zip(flow["questions"], flow["answers"]):
                    correct = answer == question.answer_index
                    label = "Correct" if correct else "Review"
                    st.write(f"**{label}:** {question.stem}")
                    st.write(f"Your answer: {question.options[answer]}")
                    if not correct:
                        st.write(f"Best answer: {question.options[question.answer_index]}")
                    st.caption(question.explanation)
        if outcome == "allow":
            st.success("No impulsivity gate was triggered for this synthetic order.")
        elif outcome == "warn":
            st.warning("Warning: review your plan carefully before placing this order.")
        else:
            cooldown = COOLDOWN_SECONDS[outcome]
            st.error(
                f"{outcome.title()}: new orders are locked for {cooldown} seconds "
                "as a cooling-off period."
            )
        if st.button("Check another trade"):
            flow.clear()
            flow["phase"] = "order"
            st.rerun()

        with st.expander("Decision explanation and optional speech"):
            if st.button("Generate educator explanation with the LLM"):
                from llm_core import generate_response

                flow["llm_explanation"] = generate_response(
                    (
                        "You are a concise Indian financial-literacy educator. "
                        f"Respond in {'Hindi' if flow['language'] == 'hi' else 'English'}. "
                        "Do not give personalized investment advice or change the gate decision."
                    ),
                    (
                        f"Explain in two short sentences why a synthetic order with "
                        f"score {flow['score']} received outcome {outcome}. "
                        f"Signals: {flow['features']}."
                    ),
                    max_new_tokens=110,
                    temperature=0.2,
                )
                st.rerun()
            if flow.get("llm_explanation"):
                st.write(flow["llm_explanation"])
                if st.button("Speak explanation"):
                    explanation_path = synthesize_audio(
                        flow["llm_explanation"],
                        flow["language"],
                        f"explanation:{flow['score']}:{outcome}:{flow['language']}",
                    )
                    st.audio(str(explanation_path), format="audio/mp3")
        with st.expander("Score details"):
            st.json(flow["features"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Test the deterministic pipeline.")
    parser.add_argument(
        "--build-audio-cache",
        action="store_true",
        help="Pre-synthesize the fixed question bank in English and Hindi.",
    )
    args, _ = parser.parse_known_args()
    if args.self_test:
        _run_self_test()
    elif args.build_audio_cache:
        count = build_audio_cache()
        print(f"Cached audio for {count} bank questions in {AUDIO_CACHE_DIR}.")
    else:
        _run_streamlit()


if __name__ == "__main__":
    main()
