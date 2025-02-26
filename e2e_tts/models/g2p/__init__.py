from . import cleaners
from .symbols import symbols

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:


def text_to_sequence(text, cleaner_names=['normalize_phonemes'], return_boundary=False):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    """

    text, boundaries = _clean_text(text, cleaner_names)
    sequence = [_symbol_to_id[w[:-1] if w.startswith("@") and w[-1].isdigit() else w] for w in text]
    
    if return_boundary:
        return sequence, boundaries
    else:
        return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    sequence = [str(_) for _ in sequence]
    sequence = ' '.join(sequence).split(str(_symbol_to_id[' ']))
    result = ' '.join([_sequence_to_symbols(s) for s in sequence])

    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception('Unknown cleaner: %s' % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s[:-1] if s.startswith("@") and s[-1].isdigit() else s] for s in symbols.split()]


def _sequence_to_symbols(sequence):
    return '_'.join([_id_to_symbol[int(s)] for s in sequence.split()])
