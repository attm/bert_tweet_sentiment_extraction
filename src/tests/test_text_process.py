import pytest
from src.data.text_process import preprocess_text


def test_preprocess_text():
    given_text = " for...the...losses. dumbface  ...him, not u. what u up to on the wknd? i wanna seeeeeee ya!"
    expected_text = "for ... the ... loss . dumbface ... him , not u. what u up to on the wknd ? i wan na seeeeeee ya !"
    actual_text = preprocess_text(given_text)
    assert actual_text == expected_text