from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

from app import _validate_professional_report_story


def _paragraph(text: str) -> Paragraph:
    return Paragraph(text, getSampleStyleSheet()["BodyText"])


def test_professional_report_quality_accepts_clean_narrative():
    story = [
        _paragraph(
            "The national flag was adopted on 15 February 1794. "
            "Its design reflects the country's political and historical development."
        )
    ]
    _validate_professional_report_story(story)


def test_professional_report_quality_rejects_ambiguous_date():
    story = [_paragraph("The flag was adopted on 15/2/1794.")]
    try:
        _validate_professional_report_story(story)
    except ValueError as exc:
        assert "ambiguous numeric calendar date" in str(exc)
    else:
        raise AssertionError("Ambiguous numeric date should be rejected")


def test_professional_report_quality_rejects_mediawiki_residue():
    story = [_paragraph("Loire {{convert|1,012|km|mi|abbr=on}}")]
    try:
        _validate_professional_report_story(story)
    except ValueError as exc:
        assert "MediaWiki template residue" in str(exc)
    else:
        raise AssertionError("MediaWiki residue should be rejected")


def test_professional_report_quality_rejects_raw_pipe_separator():
    story = [_paragraph("General: 112 | Police: 17")]
    try:
        _validate_professional_report_story(story)
    except ValueError as exc:
        assert "raw pipe separator" in str(exc)
    else:
        raise AssertionError("Raw pipe separators should be rejected")
