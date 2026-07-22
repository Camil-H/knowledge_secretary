from src.tasks.newsletter.utils import clean


def test_clean_collapses_whitespace():
    assert clean("a\n\n  b\t c") == "a b c"
    assert clean("  padded  ") == "padded"
    assert clean("") == ""
