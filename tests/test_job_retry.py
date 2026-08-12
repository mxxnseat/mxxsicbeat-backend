from app.common.job_retry import is_final_attempt


class _FakeJob:
    def __init__(self, opts: dict | None = None, attempts_made: int = 0) -> None:
        self.opts = opts if opts is not None else {}
        self.attemptsMade = attempts_made


def test_is_final_attempt_true_when_attempts_exhausted():
    job = _FakeJob(opts={"attempts": 3}, attempts_made=2)
    assert is_final_attempt(job) is True


def test_is_final_attempt_false_when_retries_remain():
    job = _FakeJob(opts={"attempts": 3}, attempts_made=1)
    assert is_final_attempt(job) is False


def test_is_final_attempt_defaults_to_true_without_opts():
    job = _FakeJob()
    assert is_final_attempt(job) is True
