import numpy as np
import re, os, glob, sys, traceback
import ast
import functools
from collections.abc import Mapping, Sequence, Set

def reentrancy_guard(method):
    """
    A decorator that prevents re-entrant calls to instance methods.
    Strips off any extra args/kwargs (e.g. Qt’s clicked(bool)) and
    always calls the wrapped method with exactly self.
    """
    flag = "_in_reentrant_section"
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # If already in the protected section, do nothing
        if getattr(self, flag, False):
            return
        setattr(self, flag, True)
        try:
            # Call only with `self`—drop all other args.
            return method(self)
        finally:
            setattr(self, flag, False)
    return wrapper


def float2str(value, length=None):
    """
    Convert float to string:
      - If only `length` is given, output a fixed-point decimal string of total width `length`.

    Args:
        value (float): number to format.
        length (int, optional): total output length including sign and decimal point.

    Returns:
        str: formatted string.
    """


    # If length is specified, use fixed-point formatting
    if length is not None:
        # enforce minimum length
        length = max(length, 5)
        # determine sign and integer part
        sign = '-' if value < 0 else ''
        abs_val = abs(value)
        int_part = str(int(abs_val))
        # calculate allowed decimal places: leave one char for decimal point
        # total = len(sign) + len(int_part) + 1 + dec_places
        dec_places = length - len(sign) - len(int_part) - 1
        if dec_places < 0:
            # no room for decimal point and decimals
            dec_places = 0
        # render as fixed-point with dec_places
        s_full = f"{value:.{dec_places}f}"
        if '.' in s_full:
            s_full = s_full.rstrip('0').rstrip('.')
        # truncate if longer
        return s_full[:length]

    # fallback: default string conversion
    return str(value)


def float2str_eng(value, length=None):
    """
    Format a float using plain decimal when 0 < |x| < 1000 and |x| > 0.001,
    otherwise use engineering notation with exponent a multiple of 3.

    Key behavior:
      - Zero -> "0".
      - Engineering mantissa lies in ENG_RANGE (default [0.1, 100); switchable to [1, 1000)).
      - If rounding would push mantissa to the upper bound, shift exponent by +3 and set
        mantissa to the lower bound.
      - `length` is the MAX total width (sign, decimal point, group separators, 'e' and exponent
        BEFORE SI replacement). We keep as many decimals as fit; trailing zeros are stripped;
        we never pad to fill length.
      - Fractional digits are grouped every 3 (internal `sep`); integer part is not grouped.
      - To avoid "0eN", if lower bound < 1, keep at least one decimal when mantissa < 1 before rounding.
      - Internal `min_length` guard prevents overly short, ambiguous outputs.
      - Only at the very end we try to replace "e±N" with an SI prefix (e.g., e12->T, e-9->n).

    Two internal constants you can tweak:
      - DEC_CAP_MAX: hard cap for decimals to try (default 18).
      - ENG_RANGE: (lower, upper) for mantissa normalization; e.g.,
          (0.1, 100.0)  # classic engineering style
          (1.0, 1000.0) # alternate style

    Returns a string.
    """

    # -------- Internal knobs (edit here) --------
    DEC_CAP_MAX = 18          # physical upper bound for decimals to try
    ENG_RANGE   = (1.0, 1000)  # change to (1.0, 1000.0) for 1–1000 mantissa
    # --------------------------------------------

    # Internal formatting guardrails
    sep = ' '     # fractional group separator (internal choice)
    min_length = 6
    if length is not None:
        try:
            length = int(length)
        except Exception:
            length = None
        else:
            if length < min_length:
                length = min_length

    # SI replacement (final-step only)
    def _si_replace(s: str) -> str:
        if 'e' not in s:
            return s
        head, _, exp = s.rpartition('e')
        try:
            k = int(exp)
        except Exception:
            return s
        si_map = {-9:'n', -6:'µ', -3:'m', 3:'k', 6:'M', 9:'G', 12:'T'}
        return head + si_map[k] if k in si_map else s

    # Helpers
    def _trim_frac(frac: str) -> str:
        return frac.rstrip('0')

    def _group_fraction(frac: str) -> str:
        if not frac:
            return ''
        return sep.join(frac[i:i+3] for i in range(0, len(frac), 3))

    def _dec_blocks_len(d: int) -> int:
        """Length contributed by d fractional digits including group separators."""
        if d <= 0:
            return 0
        seps = (d - 1) // 3  # number of separators between groups
        return d + seps

    def _cap_decimals_plain(sign_len: int, int_len: int, L) -> int:
        """Max decimals for plain mode under length L (conservative, assumes a dot is present)."""
        if L is None:
            return DEC_CAP_MAX
        budget = L - sign_len - int_len - 1  # reserve 1 for '.'
        if budget <= 0:
            return 0
        # find max d with d + (d-1)//3 <= budget
        d = min(DEC_CAP_MAX, max(0, budget))
        while d > 0 and _dec_blocks_len(d) > budget:
            d -= 1
        return max(0, d)

    def _cap_decimals_eng(sign_len: int, int_len: int, exp_tail_len: int, L) -> int:
        """Max decimals for engineering mode under length L (conservative, assumes a dot is present)."""
        if L is None:
            return DEC_CAP_MAX
        budget = L - sign_len - int_len - exp_tail_len - 1  # reserve 1 for '.'
        if budget <= 0:
            return 0
        d = min(DEC_CAP_MAX, max(0, budget))
        while d > 0 and _dec_blocks_len(d) > budget:
            d -= 1
        return max(0, d)

    # Specials
    try:
        if not np.isfinite(value):
            if np.isnan(value):
                return "nan"
            return "inf" if value > 0 else "-inf"
    except Exception:
        pass

    if value == 0.0:
        return "0"

    sign = '-' if value < 0 else ''
    vabs = np.abs(value)

    # Mode decision for plain vs engineering
    use_plain = (vabs < 1000.0) and (vabs > 0.001)

    # ---------- Plain decimal ----------
    if use_plain:
        # dynamic decimal cap from length (conservative)
        int_part_preview = str(int(np.floor(vabs)))  # preview integer part
        d_top = _cap_decimals_plain(len(sign), len(int_part_preview), length)

        for d in range(min(DEC_CAP_MAX, d_top), -1, -1):
            s = f"{vabs:.{d}f}"
            int_part, _, frac = s.partition('.')
            frac = _trim_frac(frac)
            body = int_part if not frac else int_part + '.' + _group_fraction(frac)
            cand = sign + body
            if (length is None) or (len(cand) <= length):
                return cand
        return sign + str(int(np.round(vabs)))

    # ---------- Engineering ----------
    LBOUND, UBOUND = float(ENG_RANGE[0]), float(ENG_RANGE[1])

    # initial exponent multiple of 3
    e3 = int(np.floor(np.log10(vabs) / 3.0) * 3)
    mant = vabs / (10.0 ** e3)

    # normalize mantissa to [LBOUND, UBOUND)
    if mant >= UBOUND:
        mant /= 1000.0
        e3 += 3
    elif mant < LBOUND:
        mant *= 1000.0
        e3 -= 3

    # "no 0eN" only matters when lower bound < 1 (e.g., 0.1-100)
    min_dec = 1 if (LBOUND < 1.0 and mant < 1.0) else 0

    # dynamic decimal cap from length (needs exponent length; conservative uses 'e' form)
    # Note: we check fitting against the 'e±N' form; SI replacement (shorter) happens after.
    exp_tail_len = 1 + len(str(e3)) if e3 >= 0 else 1 + 1 + len(str(-e3))  # 'e' + optional '-' + digits
    # preview int part length from current (unrounded) mantissa
    int_len_preview = len(str(int(np.floor(mant))))  # 0, 1.., up to 3 depending on ENG_RANGE
    d_top = _cap_decimals_eng(len(sign), int_len_preview, exp_tail_len, length)

    for d in range(min(DEC_CAP_MAX, d_top), min_dec - 1, -1):
        m_round = float(np.round(mant, d))
        e_round = e3

        # if rounding hits or exceeds the upper bound, shift exponent and clamp to lower bound
        if m_round >= UBOUND:
            m_round = LBOUND
            e_round = e3 + 3

        s = f"{m_round:.{d}f}"
        int_part, _, frac = s.partition('.')
        frac = _trim_frac(frac)

        if (int_part == '0') and (not frac) and (LBOUND < 1.0):
            # would become "0eN" in 0.1–100 mode; require at least one decimal
            continue

        body = int_part if not frac else int_part + '.' + _group_fraction(frac)
        cand_e = sign + body + f"e{e_round}"
        if (length is None) or (len(cand_e) <= length):
            return _si_replace(cand_e)

    # Fallback with minimal decimals (correctness first)
    d = min_dec
    m_round = float(np.round(mant, d))
    e_round = e3
    if m_round >= UBOUND:
        m_round = LBOUND
        e_round = e3 + 3
    s = f"{m_round:.{d}f}"
    int_part, _, frac = s.partition('.')
    frac = _trim_frac(frac)
    body = int_part if not frac else int_part + '.' + _group_fraction(frac)
    return _si_replace(sign + body + f"e{e_round}")

# Global namespace for safe eval
SAFE_GLOBALS = {'np': np}

def python2str(value) -> str:
    """
    Convert various Python objects to strings that can be displayed and evaluated:
      - None            -> ''
      - numbers/strings -> repr(value)
      - lists/dicts/tuples -> repr(value)
      - 1D numpy arrays with uniform spacing -> 'np.linspace' or 'np.arange'
      - other numpy arrays -> 'np.array(...)'
    """
    # Handle None
    if value is None:
        return ''
    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        # Only consider 1D arrays with more than one element
        if value.ndim == 1 and value.size > 1:
            start = value[0].item()
            diffs = np.diff(value)
            step = diffs[0].item()
            # Check for constant step size
            if np.allclose(diffs, step, rtol=1e-6, atol=0):
                stop = value[-1].item()
                count = value.size
                # Prefer linspace for clarity
                return f"np.linspace({start!r}, {stop!r}, {count})"
            # Fall back to arange if pattern matches
            end = start + step * value.size
            if np.allclose(value, np.arange(start, end, step), rtol=1e-6, atol=0):
                return f"np.arange({start!r}, {end!r}, {step!r})"
        # Fallback for non-1D or small arrays
        return f"np.array({value.tolist()!r})"
    # Fallback for any other type: use repr without newlines
    return repr(value).replace('\n', '')


def str2python(s: str):
    """
    Parse text from a QLineEdit back into a Python object:
      1) Use ast.literal_eval for literals (numbers, lists, dicts, tuples, 'None')
      2) Fallback to eval in SAFE_GLOBALS for numpy expressions
      3) Return the original string if parsing fails
    """
    s = s.strip()
    if not s:
        return None
    if s.startswith("$device:"):
        return s
    try:
        # Safely evaluate Python literals
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        try:
            # Allow numpy expressions via eval
            return eval(s, SAFE_GLOBALS, {})
        except Exception:
            # Return raw string on failure
            return s



def python2plain(x):
    """
    Minimal sanitizer:
    - Keep builtins: None, bool, int, float, str
    - numpy scalar -> item()
    - numpy ndarray -> tolist()
    - Recurse into dict/list/tuple/set to clean nested numpy
    - Anything else -> TypeError
    """
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # numpy types

    if isinstance(x, np.generic):   # numpy scalar
        return x.item()
    if isinstance(x, np.ndarray):   # numpy array
        return x.tolist()

    if isinstance(x, Mapping):
        return type(x)((python2plain(k), python2plain(v)) for k, v in x.items())

    if isinstance(x, tuple):
        return tuple(python2plain(v) for v in x)
    if isinstance(x, list):
        return [python2plain(v) for v in x]
    if isinstance(x, frozenset):
        return frozenset(python2plain(v) for v in x)
    if isinstance(x, Set):
        return [python2plain(v) for v in x]

    raise TypeError(f"python2plain: unsupported type {type(x).__name__}")



def fuzzy_search(addr: str, DIR: str, file_type: str = '.npz') -> str:
    """
    Locate a .npz file matching `addr` under various candidate locations.

    Search order:
      1. If `addr` contains a wildcard, expand it in both cwd and DIR.
      2. Otherwise, try:
         a. Exact `addr`
         b. `addr + file_type` in cwd
         c. `DIR/addr`
         d. `DIR/addr + file_type`
    Then filter for the desired extension, normalize paths, and dedupe.

    Args:
        addr (str): Base filename or pattern to search for.
        DIR (str): Directory to search in as fallback.
        file_type (str): Extension to match (default: '.npz').

    Returns:
        str: The first matching filepath, or None if none found.
    """
    # 1. Build candidate patterns
    if '*' in addr:
        # Wildcard search: expand in both cwd and DIR
        patterns = [
            addr,
            os.path.join(DIR, addr)
        ]
    else:
        # Exact and appended-extension search
        base = os.path.basename(addr)
        patterns = [
            addr,
            addr + file_type,
            os.path.join(DIR, addr),
            os.path.join(DIR, f"{base}{file_type}")
        ]

    # 2. Collect all matching paths
    matches = []
    for pat in patterns:
        if '*' in pat:
            matches.extend(glob.glob(pat))
        elif os.path.exists(pat):
            matches.append(pat)

    # 3. Filter by extension
    data_files = [p for p in matches if p.lower().endswith(file_type)]

    # 4. Normalize and deduplicate by real path
    seen = set()
    unique_files = []
    for p in data_files:
        # Normalize separators and resolve symlinks / relative segments
        real = os.path.realpath(os.path.normpath(p))
        if real not in seen:
            seen.add(real)
            unique_files.append(real)

    if not unique_files:
        print(f"No {file_type} found for '{addr}'")
        return None

    if len(unique_files) > 1:
        print("Multiple matches:", unique_files)

    # 5. Pick the first and print it
    chosen = unique_files[0]
    print(f"Loading {chosen}")
    return chosen

def log_error(e):
    """
    call in except to print more information for debugging

    try:
        pass
    except Exception as e:
        log_error(e)
    """
    print("Error:", e)
    exc_info = sys.exc_info()
    if all(exc_info):
        traceback.print_exception(*exc_info)
    else:
        print("(No active exception to trace)")

def align_to_resolution(value=None, resolution=None, allow_zero: bool = False):
    """
    Snap a number or a 'float_or_x' string to a resolution grid.

    Parameters
    ----------
    value : int|float|str|None
        A plain number, or a string in the accepted float_or_x forms
        (e.g., "123", "101-x", "x-3.5e2", "+x", etc.).
    resolution : int|float|None
        Grid step (ns in your use case). If None or 0, the value is returned unchanged.
    allow_zero : bool
        If True, do NOT modify zeros:
          - numeric 0 stays 0
          - string tokens that parse to 0 stay as-is
        If False (default), avoid producing 0 by snapping 0 to +resolution.

    Rules (when allow_zero is False)
    --------------------------------
    - value == 0              -> +resolution  (avoid zero)
    - value  > 0              -> ceil(x/res)*res
    - value  < 0              -> floor(x/res)*res
    - For strings, only the single numeric token is replaced; the rest (including 'x') is preserved.
    - Unparseable strings are returned unchanged.
    - Pure '±x' (no numeric token) is returned unchanged.

    Notes
    -----
    - Formatting tries to keep a compact numeric representation (scientific form when reasonable).
    """
    if resolution in (None, 0):
        return value

    resolution = np.abs(resolution)

    # Early return only for None/empty string (NOT numeric zero)
    if value in (None, ''):
        return value

    # -------- Numeric input --------
    if not isinstance(value, str):
        x = float(value)
        if x == 0.0:
            # Keep 0 only if explicitly allowed; otherwise snap to +resolution
            return 0 if allow_zero else int(resolution)
        elif x > 0.0:
            snapped = np.ceil(x / resolution) * resolution
        else:  # x < 0
            snapped = np.floor(x / resolution) * resolution
        return int(snapped)

    # -------- String input --------
    s = value
    m = FLOAT_OR_X_PARSING_PATTERN.fullmatch(s)
    if not m:
        # Not in the accepted forms; leave it as-is
        return s

    # Locate the single numeric token (there is at most one)
    if m.group('pure_num') is not None:
        a, b = m.span('pure_num')
    elif m.group('num_left') is not None:
        a, b = m.span('num_left')
    elif m.group('num_right') is not None:
        a, b = m.span('num_right')
    else:
        # Only ±x; no numeric token to align here
        return s

    tok = s[a:b]
    try:
        x = float(tok)
    except Exception:
        # Token not parseable as float; leave unchanged
        return s

    # Zero handling: keep or avoid based on allow_zero
    if x == 0.0:
        return s if allow_zero else (s[:a] + str(int(resolution)) + s[b:])

    # If already exactly on grid, preserve the original token
    r = x / resolution
    if np.isclose(r, np.round(r), rtol=0.0, atol=1e-12):
        return s

    # Snap away from zero
    if x > 0.0:
        snapped = float(np.ceil(r) * resolution)
    else:  # x < 0.0
        snapped = float(np.floor(r) * resolution)

    # --- Formatting: keep your original compact numeric formatting ---
    # Choose a small decimal precision k so step * 10^k is (almost) integer (cap 12)
    k = 0
    while k < 12:
        if np.isclose(resolution * (10 ** k), np.round(resolution * (10 ** k)), rtol=0.0, atol=1e-12):
            break
        k += 1

    sign = '-' if snapped < 0 else ''
    s_fix = f"{abs(snapped):.{k}f}"  # stabilize digits with fixed k decimals
    if '.' in s_fix:
        ip, fp = s_fix.split('.', 1)
    else:
        ip, fp = s_fix, ''
    fp = fp.rstrip('0')

    if fp:
        # Try AeB where A,B are integers, B != 0
        A_str = (ip + fp).lstrip('0') or '0'
        B = -len(fp)
        if A_str != '0' and B != 0:
            new_tok = f"{sign}{A_str}e{B}"
        else:
            # Fallback: plain decimal, trimmed
            new_tok = f"{snapped:.{k}f}"
            if '.' in new_tok:
                new_tok = new_tok.rstrip('0').rstrip('.')
    else:
        new_tok = f"{snapped:.{k}f}"
        if '.' in new_tok:
            new_tok = new_tok.rstrip('0').rstrip('.')

    # Replace only the numeric substring
    return s[:a] + new_tok + s[b:]

def data_x_str_validator(data_x, n_dim):
    """
    Validate that `data_x` (str or object) parses via `str2python` into:
      - a numpy.ndarray
      - with ndim == 2
      - shape[1] == n_dim
      - numeric elements (if not numeric dtype, attempt to cast to float)

    Returns:
        (ok: bool, arr_or_none: Optional[np.ndarray], err_or_none: Optional[str])
    """
    s = data_x
    try:
        obj = str2python(s) if isinstance(s, str) else s
    except Exception as e:
        return False

    if obj is None:
        return False

    if not isinstance(obj, np.ndarray):
        return False

    if obj.ndim != 2:
        return False

    if obj.shape[1] != int(n_dim):
        return False

    if len(obj)<=1:
        return False

    # Coerce to numeric if needed
    if not np.issubdtype(obj.dtype, np.number):
        try:
            obj = obj.astype(float)
        except Exception:
            return False

    return True


# Matches either:
#   • an empty string, OR
#   • a comma-separated list of 2D coordinate pairs, each in () or [] brackets,
#     where each number may be integer, decimal, or in scientific notation.
COORDINATE_LIST_PATTERN = re.compile(r"""
    ^(?:                                       # Anchor: start of string
        $                                      # 1) Empty string
      |                                        # OR
        (?:                                    # 2) One or more coordinate pairs
            [\(\[]                             #   Opening bracket "(" or "["
            \s*                                #   Optional whitespace
            [+-]?                              #   Optional plus/minus sign
            \d+                                #   One or more digits
            (?:\.\d+)?                         #   Optional decimal fraction
            (?:[eE][+-]?\d+)?                  #   Optional exponent, e.g. "e-3"
            \s*,\s*                            #   Comma separator with optional spaces
            [+-]?                              #   Optional minus sign
            \d+                                #   One or more digits
            (?:\.\d+)?                         #   Optional decimal fraction
            (?:[eE][+-]?\d+)?                  #   Optional exponent
            \s*                                #   Optional whitespace
            [\)\]]                             #   Closing bracket ")" or "]"
        )
        (?:                                    #   Additional coordinate pairs...
            \s*,\s*                            #     Comma separator
            [\(\[]\s*                          #     Opening bracket + optional spaces
            [+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?#     Number #1
            \s*,\s*                            #     Comma
            [+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?#     Number #2
            \s*[\)\]]                          #     Closing bracket + optional spaces
        )*                                     #   Zero or more more pairs
    )
    $                                          # Anchor: end of string
""", re.VERBOSE)


# Matches a single floating-point number in plain or scientific notation, with optional sign:
#   • Optional leading '+' or '-'
#   • Digits before the decimal point (optional if there’s at least one digit after the dot)
#   • A mandatory decimal point and at least one digit after it
#   • Optional exponent part, e.g. “e-10” or “E+03”
FLOAT_PATTERN = re.compile(r"""
    ^                         # start of string
    [+-]?                     # optional sign
    (?:                       # one of:
        \d*\.\d+              #   .123 or 0.123 or 12.34
      |                       # OR
        \d+\.?                #   123. or 123  (requires at least one digit)
    )
    (?:[eE][+-]?\d+)?         # optional exponent: e or E, optional sign, digits
    $                         # end of string
""", re.VERBOSE)




def _float_core_from(pattern_obj: re.Pattern) -> str:
    """
    Extract the inner body of a VERBOSE float pattern:
    - Remove the entire lines that contain '^' or '$' anchors (including comments).
    - Keep all other content (spacing/comments) intact for (?x) embedding.
    """
    core = pattern_obj.pattern
    # Remove the line starting with '^' (and its comment), in multiline mode
    core = re.sub(r'(?m)^\s*\^\s*(?:#.*)?$', '', core)
    # Remove the line containing '$' (and its comment), in multiline mode
    core = re.sub(r'(?m)^\s*\$\s*(?:#.*)?$', '', core)
    return core

def insert_mul_before_x(expr: str) -> str:
    """
    Turn 'kx' (integer k) into 'k*x' so that Python eval works.
    Does not modify spaces or signs; only inserts '*' between integer and 'x'.
    """
    return re.sub(r'(\d+)\s*x', r'\1*x', expr)


_num_core = _float_core_from(FLOAT_PATTERN)


# STRICT: final semantics for parsing/snap/logic (single numeric token only)
FLOAT_OR_X_PARSING_PATTERN = re.compile(
    r"^\s*(?:"
    # number ± (k)x      -> we still only "snap" the single number token
    r"(?P<num_left>(?x:" + _num_core + r"))\s*[+-]\s*(?:\d+\s*)?x"
    r"|"
    # (k)x ± number      -> the number on the right is the single token
    r"(?:\d+\s*)?x\s*[+-]\s*(?P<num_right>(?x:" + _num_core + r"))"
    r"|"
    # pure ±(k)x         -> no numeric token to align
    r"[+-]?(?:\d+\s*)?x"
    r"|"
    # pure number
    r"(?P<pure_num>(?x:" + _num_core + r"))"
    r")\s*$"
)


# UI validator (loose): accept pure number, number±(k)x, ±(k)x±number
FLOAT_OR_X_PATTERN = re.compile(
    r"^\s*(?:"
    # number [± (k)x]
    r"(?:(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)(?:\s*[+\-]\s*(?:\d+\s*)?x)?"
    r"|"
    # ±(k)x [± number]
    r"[+\-]?(?:\d+\s*)?x(?:\s*[+\-]\s*(?:(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?))?"
    r")\s*$",
    re.VERBOSE
)


