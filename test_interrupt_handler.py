"""
Lightweight behavioural checks for InterruptHandler.

Run with:
    python test_interrupt_handler.py
"""

import asyncio
import sys

from interrupt_handler import InterruptHandler


def build_cases():
    """
    Each entry describes one test probe for the handler.

    Fields:
        id           : short numeric id
        title        : human label
        text         : user transcript
        conf         : STT confidence
        lang         : language code
        speaking     : whether agent is currently talking
        expect       : expected InterruptHandler 'action'
    """
    return [
        # --- Core scenarios from the spec ---
        {
            "id": 1,
            "title": "Simple English filler while agent speaks",
            "text": "umm",
            "conf": 0.85,
            "lang": "en",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 2,
            "title": "Explicit interruption phrase",
            "text": "wait one second",
            "conf": 0.92,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 3,
            "title": "Filler while agent is silent",
            "text": "umm",
            "conf": 0.88,
            "lang": "en",
            "speaking": False,
            "expect": "REGISTER",
        },
        {
            "id": 4,
            "title": "Filler plus stop command",
            "text": "umm okay stop",
            "conf": 0.87,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 5,
            "title": "Low-confidence background noise",
            "text": "hmm yeah",
            "conf": 0.35,
            "lang": "en",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 6,
            "title": "Hindi filler during agent speech",
            "text": "haan",
            "conf": 0.88,
            "lang": "hi",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 7,
            "title": "Custom runtime filler (\"basically\")",
            "text": "basically",
            "conf": 0.85,
            "lang": "en",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 8,
            "title": "Hindi hard stop command",
            "text": "ruko",
            "conf": 0.90,
            "lang": "hi",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 9,
            "title": "Empty transcript",
            "text": "",
            "conf": 0.70,
            "lang": "en",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 10,
            "title": "Normal question while agent quiet",
            "text": "what is the weather",
            "conf": 0.95,
            "lang": "en",
            "speaking": False,
            "expect": "REGISTER",
        },
        {
            "id": 11,
            "title": "Sequence of multiple fillers",
            "text": "uh umm hmm",
            "conf": 0.85,
            "lang": "en",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 12,
            "title": "Question that should cut in",
            "text": "how does that work",
            "conf": 0.92,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },

        # --- Additional coverage cases (13–25) ---
        {
            "id": 13,
            "title": "Long stretched hesitation sound",
            "text": "hmmmmmmmmmmm",
            "conf": 0.90,
            "lang": "en",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 14,
            "title": "Hesitation then meaningful phrase",
            "text": "uhh okay I think",
            "conf": 0.90,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 15,
            "title": "Spanish filler word while speaking",
            "text": "este",
            "conf": 0.80,
            "lang": "es",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 16,
            "title": "French filler while speaking",
            "text": "euh",
            "conf": 0.85,
            "lang": "fr",
            "speaking": True,
            "expect": "IGNORE",
        },
        {
            "id": 17,
            "title": "French filler plus English stop",
            "text": "euh stop",
            "conf": 0.90,
            "lang": "fr",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 18,
            "title": "Very low-confidence real sentence, silent agent",
            "text": "hello can you hear me",
            "conf": 0.30,
            "lang": "en",
            "speaking": False,
            "expect": "IGNORE",
        },
        {
            "id": 19,
            "title": "High-confidence filler while agent quiet",
            "text": "hmm hmm",
            "conf": 0.90,
            "lang": "en",
            "speaking": False,
            "expect": "REGISTER",
        },
        {
            "id": 20,
            "title": "Short affirmative while agent speaks",
            "text": "yes",
            "conf": 0.90,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 21,
            "title": "Gibberish token while agent speaks",
            "text": "asdfghjk",
            "conf": 0.90,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 22,
            "title": "Polite follow-up question while speaking",
            "text": "can you explain that again",
            "conf": 0.95,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 23,
            "title": "Spanish request while agent quiet",
            "text": "puedes repetir eso",
            "conf": 0.93,
            "lang": "es",
            "speaking": False,
            "expect": "REGISTER",
        },
        {
            "id": 24,
            "title": "Variant of hold-on command",
            "text": "hold on for a second",
            "conf": 0.90,
            "lang": "en",
            "speaking": True,
            "expect": "INTERRUPT",
        },
        {
            "id": 25,
            "title": "Hindi filler combination while quiet",
            "text": "hmm haan theek",
            "conf": 0.90,
            "lang": "hi",
            "speaking": False,
            "expect": "REGISTER",
        },
    ]


async def main() -> int:
    print("\n================= INTERRUPT HANDLER CHECK =================\n")

    handler = InterruptHandler(
        confidence_threshold=0.6,
        enable_contextual_analysis=True,
        enable_multi_language=True,
        log_events=False,  # keep test output focused
    )

    # one-time dynamic filler registration
    handler.add_custom_fillers("en", ["basically"])

    cases = build_cases()
    success_flags = []

    # header row
    print(f"{'ID':>3} | {'RESULT':^7} | {'EXPECTED':^9} | {'ACTUAL':^9} | TEXT")
    print("-" * 72)

    for case in cases:
        handler.set_agent_speaking(case["speaking"])
        res = await handler.process_speech_event(
            transcript=case["text"],
            confidence=case["conf"],
            language=case["lang"],
        )
        action = res["action"]
        ok = (action == case["expect"])
        success_flags.append(ok)

        status = "OK" if ok else "MISMATCH"
        print(
            f"{case['id']:>3} | {status:^7} | {case['expect']:^9} | {action:^9} | {case['title']}"
        )

    # print handler’s own statistics at the end (separate from pass/fail)
    print("\n---- Internal statistics from InterruptHandler ----")
    handler.print_summary()

    total = len(cases)
    passed = sum(1 for x in success_flags if x)
    failed = total - passed

    print("Overall test result:")
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
