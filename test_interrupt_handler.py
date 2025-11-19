"""
Comprehensive Test Suite for InterruptHandler.

Run:
    python test_interrupt_handler.py
"""

import asyncio
import sys
from interrupt_handler import InterruptHandler


class TestResults:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.tests = []

    def add_result(self, name: str, expected: str, actual: str, passed: bool) -> None:
        self.tests.append(
            {"name": name, "expected": expected, "actual": actual, "passed": passed}
        )
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def print_summary(self) -> bool:
        print("\n" + "-" * 70)
        print("TEST SUITE SUMMARY")
        print("-" * 70)
        for t in self.tests:
            status = "PASS" if t["passed"] else "FAIL"
            print(f"{status} | {t['name']}")
            print(f"      Expected: {t['expected']}  Got: {t['actual']}")
        print("-" * 70)
        total = len(self.tests)
        print(f"Total : {total} tests")
        print(f"Passed: {self.passed} ({(self.passed / max(total, 1)) * 100:.1f}%)")
        print(f"Failed: {self.failed}")
        print("-" * 70 + "\n")
        return self.failed == 0


async def run_tests() -> int:
    results = TestResults()

    print("\n" + "-" * 70)
    print("INTERRUPT HANDLER TESTS")
    print("-" * 70 + "\n")

    handler = InterruptHandler(
        confidence_threshold=0.6,
        enable_contextual_analysis=True,
        enable_multi_language=True,
        log_events=False,
    )

    # 1: User filler while agent speaks
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("umm", 0.85, "en")
    expected = "IGNORE"
    actual = res["action"]
    results.add_result("Filler while agent speaking", expected, actual, actual == expected)
    print("Test 1:", actual, res["reason"])

    # 2: Real interruption while agent speaks
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("wait one second", 0.92, "en")
    expected = "INTERRUPT"
    actual = res["action"]
    results.add_result("Priority command interruption", expected, actual, actual == expected)
    print("Test 2:", actual, res["reason"])

    # 3: Filler while agent quiet
    handler.set_agent_speaking(False)
    res = await handler.process_speech_event("umm", 0.88, "en")
    expected = "REGISTER"
    actual = res["action"]
    results.add_result("Filler while quiet", expected, actual, actual == expected)
    print("Test 3:", actual, res["reason"])

    # 4: Mixed filler + command
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("umm okay stop", 0.87, "en")
    expected = "INTERRUPT"
    actual = res["action"]
    results.add_result("Mixed filler + command", expected, actual, actual == expected)
    print("Test 4:", actual, res["reason"])

    # 5: Low confidence noise
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("hmm yeah", 0.35, "en")
    expected = "IGNORE"
    actual = res["action"]
    results.add_result("Low confidence noise", expected, actual, actual == expected)
    print("Test 5:", actual, res["reason"])

    # 6: Hindi filler
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("haan", 0.88, "hi")
    expected = "IGNORE"
    actual = res["action"]
    results.add_result("Hindi filler", expected, actual, actual == expected)
    print("Test 6:", actual, res["reason"])

    # 7: Dynamic filler addition
    handler.add_custom_fillers("en", ["basically"])
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("basically", 0.85, "en")
    expected = "IGNORE"
    actual = res["action"]
    results.add_result("Dynamic filler add", expected, actual, actual == expected)
    print("Test 7:", actual, res["reason"])

    # 8: Priority Hindi command
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("ruko", 0.9, "hi")
    expected = "INTERRUPT"
    actual = res["action"]
    results.add_result("Hindi priority command", expected, actual, actual == expected)
    print("Test 8:", actual, res["reason"])

    # 9: Empty input
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("", 0.7, "en")
    expected = "IGNORE"
    actual = res["action"]
    results.add_result("Empty input", expected, actual, actual == expected)
    print("Test 9:", actual, res["reason"])

    # 10: Meaningful speech while quiet
    handler.set_agent_speaking(False)
    res = await handler.process_speech_event("what is the weather", 0.95, "en")
    expected = "REGISTER"
    actual = res["action"]
    results.add_result("Question while quiet", expected, actual, actual == expected)
    print("Test 10:", actual, res["reason"])

    # 11: Multiple fillers sequence
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("uh umm hmm", 0.85, "en")
    expected = "IGNORE"
    actual = res["action"]
    results.add_result("Multiple fillers", expected, actual, actual == expected)
    print("Test 11:", actual, res["reason"])

    # 12: Question while agent speaking
    handler.set_agent_speaking(True)
    res = await handler.process_speech_event("how does that work", 0.92, "en")
    expected = "INTERRUPT"
    actual = res["action"]
    results.add_result("Question interruption", expected, actual, actual == expected)
    print("Test 12:", actual, res["reason"])

    handler.print_summary()
    all_passed = results.print_summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_tests())
    sys.exit(exit_code)
