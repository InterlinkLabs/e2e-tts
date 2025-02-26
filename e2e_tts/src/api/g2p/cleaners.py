"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "basic_cleaners" if you do not want to transliterate (in this case, list phonemes converted).
  2. "normalize_phonemes" for normalize phoneme text
"""

import re
from typing import Union
from .g2p import normalize_phonemes

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def basic_cleaners(text: str) -> str:
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = text.upper()
    text = re.sub(_whitespace_re, ' ', text)

    return text, None, None


def normalize_phonemes(text: str) -> Union[str, list]:
    # Function to clean and normalize phoneme strings
    text = text.lower()
    text = re.sub(_whitespace_re, " ", text)
    text, boundaries = normalize_phonemes(text, is_training=False)

    return text, boundaries
