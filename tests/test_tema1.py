import pytest
from pydantic import ValidationError

from Tema1 import Student, StudentValidator, load_student


def test_student_json_is_loaded_and_validated():
    student = load_student()

    assert isinstance(student, StudentValidator)
    assert student.nume == "Ioana Tocaciu"
    assert student.varsta == 25
    assert student.email == "tocaciu_ioana@yahoo.com"


def test_student_alias_uses_student_validator():
    assert Student is StudentValidator


def test_invalid_email_is_rejected():
    with pytest.raises(ValidationError):
        Student(nume="Ana Popescu", varsta=20, email="not-an-email")


@pytest.mark.parametrize("age", [0, -1])
def test_non_positive_age_is_rejected(age):
    with pytest.raises(ValidationError):
        Student(nume="Ioana Tocaciu", varsta=age, email="tocaciu_ioana@yahoo.com")
