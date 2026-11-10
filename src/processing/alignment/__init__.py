"""Sentence and word alignment; sentence-parser imports remain compatible."""
from .sentences import parse_alignment, parse_indices, validate_alignment

__all__ = ['parse_alignment', 'parse_indices', 'validate_alignment']
