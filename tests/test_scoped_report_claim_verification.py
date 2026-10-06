from pathlib import Path


WRITER_SOURCE = Path("src/flag_recognition/report_writer.py").read_text(encoding="utf-8")


def test_scoped_review_requires_claim_level_web_verification():
    normalized = " ".join(WRITER_SOURCE.split()).casefold()
    assert "verify every concrete factual claim" in normalized
    assert "prefer primary or official sources" in normalized
    assert "remove or qualify any claim you cannot verify" in normalized
