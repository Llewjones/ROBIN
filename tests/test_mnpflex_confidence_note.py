"""MNP-Flex states its high-confidence score under its own section.

MNP-Flex has no High/Medium/Low tiers, so it is absent from
CLASSIFIER_CONFIDENCE_THRESHOLDS and from the note under the methylation
classification table. The cut-off is stated in the MNP-Flex section instead.
"""

import inspect

from robin.reporting.sections.mnpflex import MNPFLEX_HIGH_CONFIDENCE_SCORE


def test_high_confidence_score():
    assert MNPFLEX_HIGH_CONFIDENCE_SCORE == 0.30


def test_note_is_rendered_in_the_mnpflex_section():
    import robin.reporting.sections.mnpflex as section

    source = inspect.getsource(section)
    assert "High Confidence Result" in source
    assert "MNPFLEX_HIGH_CONFIDENCE_SCORE" in source


def test_mnpflex_is_not_in_the_classifier_threshold_table():
    """It has no tiers; adding it there would invent Medium/Low cut-offs."""
    from robin.classification_config import CLASSIFIER_CONFIDENCE_THRESHOLDS

    assert not any(
        "mnpflex" in key.casefold() for key in CLASSIFIER_CONFIDENCE_THRESHOLDS
    )
