import json
from pathlib import Path

from pydantic import BaseModel, EmailStr, Field


class StudentValidator(BaseModel):
    nume: str
    varsta: int = Field(gt=0)
    email: EmailStr


Student = StudentValidator


def load_student(path: Path | None = None) -> StudentValidator:
    json_path = path or Path(__file__).with_name("student.json")
    with json_path.open(encoding="utf-8") as student_file:
        data = json.load(student_file)
    return Student(**data)


if __name__ == "__main__":
    print(load_student())