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

def align_to_resolution(value=None, resolution=None, allow_any: bool = True):
    """
    Round to the nearest multiple of `resolution` (ties half away from zero).

    Rules
    -----
    - If `value` is numeric or a pure-number string:
        * Respect `allow_any`: when False, the final snapped result must be > 0.
    - If `value` is a string containing 'x':
        * ALWAYS behave as allow_any=True (negatives/zero allowed).
        * If the snapped numeric token becomes 0, remove the adjacent '+/- 0' term
          (e.g., 'x-1' -> 'x' when resolution=12).

    - Only the single numeric token is modified for strings; others (including 'x') are preserved.
    - Unparseable strings (w.r.t. FLOAT_OR_X_PARSING_PATTERN) are returned unchanged.
    """
    EPS = 1e-12

    if resolution in (None, 0):
        return value
    res = float(abs(resolution))

    # ---------- helpers ----------
    def is_zero(y: float) -> bool:
        return np.isclose(y, 0.0, rtol=0.0, atol=EPS)

    def round_half_away(q: float) -> float:
        """Nearest-integer rounding with ties away from zero."""
        return np.floor(q + 0.5) if q >= 0 else np.ceil(q - 0.5)

    def snap(x: float, enforce_positive: bool) -> float:
        """Snap x to nearest grid, optionally enforcing strictly positive result."""
        y = float(round_half_away(x / res) * res)
        if enforce_positive and (y <= 0 or is_zero(y)):
            y = res
        # collapse -0.0
        if is_zero(y):
            y = 0.0
        return y

    def decimals_for(step: float) -> int:
        """Small k so that step*10^k is (almost) integer; cap at 12."""
        k = 0
        while k < 12 and not np.isclose(step*(10**k), round(step*(10**k)), rtol=0.0, atol=EPS):
            k += 1
        return k

    def fmt(y: float) -> str:
        """Compact formatting aligned to resolution granularity."""
        if is_zero(y):
            return "0"
        k = decimals_for(res)
        s = f"{y:.{k}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s

    def simplify_zero_term(s: str, a: int, b: int, branch: str) -> str:
        """
        Drop the '+/- 0' around the numeric token.
        - num_right:  (k)x  ±  0  -> remove operator and trailing zero
        - num_left:   0  ±  (k)x  -> remove leading zero and operator
        - pure_num:   -> "0"
        """
        if branch == 'num_right':
            prefix = s[:a]
            m = re.search(r'\s*[+\-]\s*$', prefix)
            if m:
                return (prefix[:m.start()] + s[b:]).strip()
        if branch == 'num_left':
            suffix = s[b:]
            m = re.match(r'^\s*[+\-]\s*', suffix)
            if m:
                return (s[:a] + suffix[m.end():]).strip()
        return "0"

    # ---------- numeric input ----------
    if not isinstance(value, str):
        y = snap(float(value), enforce_positive=not allow_any)
        yi = round(y)
        return int(yi) if np.isclose(y, yi, rtol=0.0, atol=EPS) else y

    # ---------- string input ----------
    s = value
    m = FLOAT_OR_X_PARSING_PATTERN.fullmatch(s)
    if not m:
        return s  # Not an accepted form

    # locate the single numeric token and the matching branch
    if m.group('pure_num') is not None:
        a, b = m.span('pure_num'); branch = 'pure_num'
    elif m.group('num_left') is not None:
        a, b = m.span('num_left'); branch = 'num_left'
    elif m.group('num_right') is not None:
        a, b = m.span('num_right'); branch = 'num_right'
    else:
        return s  # pure ±(k)x -> nothing to align

    tok = s[a:b]
    try:
        x = float(tok)
    except Exception:
        return s

    # Decide positivity enforcement:
    # - For strings with 'x' (num_left/num_right): ALWAYS behave like allow_any=True.
    # - For pure numeric strings: follow allow_any.
    enforce_positive = (branch == 'pure_num') and (not allow_any)

    # already on grid?
    on_grid = np.isclose(x / res, round(x / res), rtol=0.0, atol=EPS)

    if on_grid:
        if branch != 'pure_num':
            # x-containing: drop ±0, otherwise keep as-is
            return simplify_zero_term(s, a, b, branch) if is_zero(x) else s
        # pure numeric
        if enforce_positive and x <= 0:
            return fmt(res)
        return s

    # snap and maybe simplify ±0 for x-containing strings
    y = snap(x, enforce_positive=enforce_positive)
    if (branch != 'pure_num') and is_zero(y):
        return simplify_zero_term(s, a, b, branch)

    return s[:a] + fmt(y) + s[b:]

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

# UI validator (loose): accept pure number (now with optional leading sign), number±(k)x, ±(k)x±number
FLOAT_OR_X_PATTERN = re.compile(
    r"^\s*(?:"
    # [±]number [± (k)x]   <-- add optional leading sign here to accept pure negatives
    r"[+\-]?(?:(?:\d*\.\d+|\d+\.?)(?:[eE][+\-]?\d+)?)(?:\s*[+\-]\s*(?:\d+\s*)?x)?"
    r"|"
    # ±(k)x [± number]     (keep as-is so we don't accept things like 'x--3' unless you want to)
    r"[+\-]?(?:\d+\s*)?x(?:\s*[+\-]\s*(?:(?:\d*\.\d+|\d+\.?)(?:[eE][+\-]?\d+)?))?"
    r")\s*$",
    re.VERBOSE
)



# key must start at column 0: avoids catching "'apd_signal': ..." inside dict
_KEYLINE = re.compile(r'^([A-Za-z_]\w*)\s*:\s*(.*)$')

def parse_kv_blocks_strict(text: str):
    """
    Parse blocks of:
      key: value
    where value can span multiple lines (continuations do NOT start with key: at col0).

    Returns:
      (params: dict, err: str|None)
    """
    params = {}
    cur_key = None
    cur_lines = []
    cur_start_line = None

    def _flush():
        nonlocal cur_key, cur_lines, cur_start_line
        if cur_key is None:
            return None

        blob = "\n".join(cur_lines).strip()
        val = str2python(blob)

        params[cur_key] = val
        cur_key = None
        cur_lines = []
        cur_start_line = None
        return None

    for lineno, raw in enumerate((text or "").splitlines(), 1):
        line = raw.rstrip()
        if not line.strip():
            continue

        m = _KEYLINE.match(line)
        if m:
            # finish previous block
            err = _flush()
            if err:
                return None, err

            cur_key = m.group(1)
            cur_start_line = lineno
            cur_lines = [m.group(2)]
        else:
            if cur_key is None:
                return None, f"Line {lineno}: expected 'key: value' starting at column 0."
            cur_lines.append(line)

    # flush last
    err = _flush()
    if err:
        return None, err

    return params, None