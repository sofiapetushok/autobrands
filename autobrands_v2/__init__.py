"""Independent, opt-in AutoBrands V2 implementation."""

from .matching import Matcher
from .normalize import normalize_text

__all__ = ["Matcher", "normalize_text"]
