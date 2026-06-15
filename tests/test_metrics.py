from benchmark.metrics import is_refusal, summarize


def test_refusal_detection() -> None:
    assert is_refusal("I cannot help with that request.")


def test_summary_metrics() -> None:
    rows = [
        {
            "attack_type": "jailbreak",
            "refused": True,
            "harmful_compliance": False,
            "response_tokens": 7,
            "total_ms": "100.0",
        },
        {
            "attack_type": "benign",
            "refused": False,
            "harmful_compliance": False,
            "response_tokens": 20,
            "total_ms": "200.0",
        },
    ]
    summary = summarize(rows)
    assert summary["refusal_rate"] == 0.5
    assert summary["average_latency_ms"] == 150.0
