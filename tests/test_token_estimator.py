from repo_rivet.memory.token_calibrator import UsageCalibrator
from repo_rivet.memory.token_estimator import (
    ApproximateTokenEstimator,
    CalibratedTokenEstimator,
    ProviderTokenizerEstimator,
    create_token_estimator,
)


def test_provider_estimator_uses_known_bpe_encoding() -> None:
    estimator = ProviderTokenizerEstimator.create(
        "custom-model",
        encoding_name="cl100k_base",
    )
    text = "hello world 你好"

    assert estimator.estimate_text(text) == len(estimator.encoding.encode(text))
    assert estimator.name == "tokenizer:cl100k_base"


def test_approximate_estimator_is_conservative_for_cjk_json_and_opaque_text() -> None:
    estimator = ApproximateTokenEstimator(safety_factor=1.20)

    assert estimator.estimate_text("中文测试") >= 5
    assert estimator.estimate_text("a" * 60, kind="json") >= 72
    assert estimator.estimate_text("ordinary English sentence") >= 7


def test_request_estimate_includes_tools_tool_calls_ids_and_wrapping() -> None:
    estimator = ApproximateTokenEstimator()
    plain = estimator.estimate_request([{"role": "user", "content": "task"}], [])
    complete = estimator.estimate_request(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "arguments": '{"path":"app.py"}'}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ],
        [{"type": "function", "function": {"name": "read_file"}}],
    )

    assert complete > plain


def test_unknown_model_and_encoding_fall_back_without_blocking_agent() -> None:
    estimator = create_token_estimator(
        model="repo-rivet-unknown-model",
        tokenizer_encoding="not-a-real-encoding",
    )

    assert isinstance(estimator, ApproximateTokenEstimator)


def test_calibrated_estimator_applies_feedback_factor() -> None:
    base = ApproximateTokenEstimator(safety_factor=1.0)
    calibrator = UsageCalibrator(default_factor=1.25)
    estimator = CalibratedTokenEstimator(base, calibrator)
    messages = [{"role": "user", "content": "hello"}]

    raw = estimator.raw_estimate_request(messages, [])

    assert estimator.estimate_request(messages, []) >= raw * 1.25
