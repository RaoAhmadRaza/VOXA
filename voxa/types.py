"""Pure data types for Voxa results. Stdlib only — no engine import.

Kept separate from core.py so formats and their tests never pull in the model
stack (voxa.engine / ctranslate2 / av). Data is just data.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None


@dataclass
class Info:
    language: str
    language_probability: float
    duration: float


@dataclass
class TranscriptResult:
    segments: List[Segment] = field(default_factory=list)
    info: Optional[Info] = None
