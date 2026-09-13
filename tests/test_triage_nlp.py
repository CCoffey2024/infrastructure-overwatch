"""These tests intentionally never construct a NoteTriageClassifier or call
train_note_triage: both require downloading real pretrained weights
(transformers/peft), which is out of scope for CI. They only cover the
pure-Python parts, and that importing the module doesn't itself require the
`nlp` extra to be installed (every heavy dependency is lazy-imported).
"""

from infrastructure_overwatch.triage_nlp import NOTE_CATEGORIES, TriageResult


def test_note_categories_are_the_four_documented_categories():
    assert NOTE_CATEGORIES == (
        "confirmed_threat",
        "false_alarm",
        "sensor_or_equipment_issue",
        "needs_more_information",
    )


def test_triage_result_fields():
    result = TriageResult(text="wind-blown debris, not a real target", category="false_alarm", confidence=0.87)
    assert result.text == "wind-blown debris, not a real target"
    assert result.category == "false_alarm"
    assert result.confidence == 0.87
