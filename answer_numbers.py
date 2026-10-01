#!/usr/bin/env python3
r"""How a number is read from a free-text answer, and how far it lies from an option.

Every analysis reads numbers through this module, so a correction reaches every table at once. It
needs only the standard library.

* ``numbers(text)``: every number in an answer, in order, as ``(value, per_cent)`` pairs.
  - Scientific notation is one number even when written with a multiplication sign or superscripts:
    ``7.29 × 10⁷``, ``2.1 x 10^-26`` and ``10⁻⁵`` read as 7.29e7, 2.1e-26 and 1e-5.
  - Digits inside an identifier are not numbers: ``CD14``, ``m6A``, ``H3K27ac``, ``at2759``, ``v1.5``.
  - Nor is a list marker: the ``4.`` or ``2)`` that opens a line of text.
  - A trailing per cent sign is kept, so ``0.29%`` reads as a per cent.
* ``graded(reply, ideal, tolerance)``: the tolerance. Is the last number in ``reply`` within a
  relative ``tolerance`` of the key? Against a key of zero the tolerance is absolute. The bound is
  inclusive, so a number exactly 5% from the key is within 5%.
* ``option_distance(a, v)`` and ``nearest_ranks(a, values)``: the nearest-option rule's distance,
  d(a, v) = |a - v| / (|a| + |v|). For values of one sign it equals tanh(|log|a| - log|v|| / 2), and
  options are compared on that log scale, so the comparison does not saturate in floating point for
  an answer many orders of magnitude beyond an option. An option of the other sign, or zero against
  a non-zero value, is at distance 1.

What these readings replace, and what each change moves:
* The previous reader took any run of digits, so the 14 of ``CD14`` or the 6 of ``m6A`` could be
  "the last number".
* It split ``7.29 × 10⁷`` into 7.29 and 10.
* It read a list marker, the ``4.`` of "4. Extract the log2 fold change", as a number.
* It never saw a per cent sign, so the per-cent normalisation applied in one direction only.
* It compared ``|a - y| <= 0.05 |y|`` in floating point, which rejects some answers exactly 5% off
  (0.076 against 0.08).
* The distance ``|a - v| / (|a| + |v|)`` rounds to exactly 1.0 once an answer is about 10^16 times an
  option, so a miss far beyond an extreme key tied four ways instead of going to the key.
"""
from __future__ import annotations

import math
import re

__all__ = ["NUMBER", "normalise", "as_number", "numbers", "last_number", "first_number", "graded",
           "option_distance", "nearest_ranks"]

_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPER_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

# a mantissa times ten to a power: 7.29 × 10⁷, 2.1 x 10^-26, 3·10^(4), 1.2*10**-3, 5 \times 10^{-4}
_TIMES_TEN = re.compile(
    r"(?P<m>(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?<![\w.])[-+]?\.\d+)"
    r"\s*(?:[x×✕·*]|\\times|\\cdot)\s*10\s*"
    r"(?:(?:\^|\*\*)\s*[({]?\s*(?P<e1>[-+]?\s*\d+)\s*[)}]?|(?P<e2>[⁻⁺]?[" + _SUPER_DIGITS + r"]+))")
# ten to a power with no mantissa: 10⁻⁵, 10^-5 (but not the number 105)
_BARE_TEN = re.compile(
    r"(?<![\w.])10\s*(?:(?:\^|\*\*)\s*[({]?\s*(?P<e1>[-+]?\s*\d+)\s*[)}]?|(?P<e2>[⁻⁺]?[" + _SUPER_DIGITS + r"]+))")

# a list marker: one or two digits and "." or ")" opening a line, with text after them on that line
_LIST_MARKER = re.compile(r"(?m)^([ \t]*(?:[-*•][ \t]*)?\(?)(\d{1,2})(?=[.)][ \t]+\S)")

# One number: a sign, thousands separators, decimals, an exponent, and an optional per cent sign; never
# a run of digits glued to a letter, an underscore or another number's decimal point.
NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s?%)?"
    r"(?![A-Za-z0-9_])")


def _exponent(match):
    text = (match.group("e1") or match.group("e2") or "").translate(_SUPER)
    return re.sub(r"\s+", "", text)


def normalise(text):
    """``text`` with scientific notation rewritten as ``<mantissa>e<exponent>``, the Unicode minus sign
    as ``-``, and a list marker's digits masked. The en dash, which BixBench's keys use for ranges, is
    left alone."""
    text = str(text or "").replace("−", "-")
    text = _LIST_MARKER.sub(lambda m: m.group(1) + "#" * len(m.group(2)), text)
    text = _TIMES_TEN.sub(lambda m: f"{m.group('m').replace(',', '')}e{_exponent(m)}", text)
    text = _BARE_TEN.sub(lambda m: f"1e{_exponent(m)}", text)
    return text


def as_number(text):
    """``(value, per_cent)`` for a string that is one number, or None. Per cent is read as written."""
    if text is None:
        return None
    cleaned = normalise(str(text)).strip().replace(",", "")
    percent = cleaned.endswith("%")
    cleaned = cleaned.rstrip("%").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return (value, percent)


def numbers(text):
    """Every number in ``text``, in order, as ``(value, per_cent)`` pairs."""
    out = []
    for token in NUMBER.findall(normalise(text)):
        got = as_number(token)
        if got is not None:
            out.append(got)
    return out


def _against(got, key):
    """``got``'s value normalised to ``key``'s writing: a per cent on one side only is scaled when
    the other side's value is a fraction (at most 1 in size)."""
    (value, got_pct), (key_value, key_pct) = got, key
    if key_pct and not got_pct and abs(value) <= 1.0:
        value *= 100.0
    if got_pct and not key_pct and abs(key_value) <= 1.0:
        key_value *= 100.0
    return value, key_value


def last_number(answer, key):
    """The number the tolerance reads from ``answer`` (its last), normalised to ``key``'s per-cent
    writing; None when either has none."""
    found, want = numbers(answer), as_number(key)
    if not found or want is None:
        return None
    return _against(found[-1], want)[0]


def first_number(answer, key):
    """As ``last_number``, reading the answer's first number instead."""
    found, want = numbers(answer), as_number(key)
    if not found or want is None:
        return None
    return _against(found[0], want)[0]


def within(value, key_value, tolerance):
    """Is ``value`` within ``tolerance`` of ``key_value``? Relative, except absolute against a key of
    exactly zero, where a relative band is either empty or everything; inclusive at the bound."""
    slack = 1 + 1e-9
    if key_value == 0.0:
        return abs(value) <= tolerance * slack
    return abs(value - key_value) <= tolerance * abs(key_value) * slack


def graded(reply, ideal, tolerance, pick="last"):
    """Is the ``pick`` ("last" or "first") number in ``reply`` within ``tolerance`` of ``ideal``?
    None when ``ideal`` is not a number; False when the reply holds none."""
    key = as_number(ideal)
    if key is None:
        return None
    found = numbers(reply)
    if not found:
        return False
    value, key_value = _against(found[-1] if pick == "last" else found[0], key)
    return within(value, key_value, tolerance)


def _log_gap(a, v):
    """|log|a| - log|v||: 0 when both are zero, infinite when their signs differ or one is zero."""
    if a == 0 and v == 0:
        return 0.0
    if a == 0 or v == 0 or (a < 0) != (v < 0):
        return math.inf
    return abs(math.log(abs(a)) - math.log(abs(v)))


def option_distance(a, v):
    """d(a, v) = |a - v| / (|a| + |v|), which is tanh(|log|a| - log|v|| / 2) for values of one sign
    and 1 for values of different signs (0 when both are zero)."""
    gap = _log_gap(a, v)
    return 1.0 if gap == math.inf else math.tanh(gap / 2)


def nearest_ranks(a, values):
    """Indices of the values nearest ``a`` under d, ties included. Values of ``a``'s sign are compared
    on the log scale, so an answer 10^20 times beyond an extreme option still goes to it; when no
    value shares ``a``'s sign every value is at distance 1 and all tie."""
    gaps = [_log_gap(a, v) for v in values]
    best = min(gaps)
    if best == math.inf:
        return list(range(len(values)))
    return [i for i, g in enumerate(gaps) if math.isclose(g, best, rel_tol=1e-9, abs_tol=1e-12)]
