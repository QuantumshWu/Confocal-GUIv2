import os
import sys
import time
from abc import ABC, abstractmethod
import types

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoLocator, ScalarFormatter
from matplotlib.widgets import RectangleSelector
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.axes_grid1 import make_axes_locatable, Divider, Size
from matplotlib import font_manager as fm

import numpy as np
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning
from IPython import get_ipython
from IPython.display import display, HTML


font_path  = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'helper', 'helvetica-light-587ebe5a59211.ttf'))
font_name = None
if os.path.exists(font_path):
    try:
        fm.fontManager.addfont(font_path)
    except Exception:
        pass
    font_name = fm.FontProperties(fname=font_path).get_name()

_new_black = 'black'

params_widget = {
    'axes.labelsize': 7.5,
    'legend.fontsize': 6.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'figure.figsize': [700/300, 500/300],
    'lines.linewidth': 1,
    'scatter.edgecolors':  _new_black,
    'legend.numpoints': 1,
    'lines.markersize': 2,
    'ytick.major.size': 1.5,  # major tick size in points
    'ytick.major.width': 0.4,  # major tick width in points
    'xtick.major.size': 1.5,  # major tick size in points
    'xtick.major.width': 0.4,  # major tick width in points
    'axes.linewidth': 0.4,
    'figure.subplot.left': 0,
    'figure.subplot.right': 1,
    'figure.subplot.bottom': 0,
    'figure.subplot.top': 1,
    'axes.titlepad':       1.5,
    'xtick.major.pad':     1.5,
    'ytick.major.pad':     1.5,
    'axes.labelpad':      1.5,
    'grid.linestyle': '--',
    'axes.grid': False,
    'text.usetex': False,
    'font.family' : 'sans-serif',
    'font.sans-serif' : ([font_name,] if font_name else []) + ['Arial',],
    "xtick.direction": "in",
    "ytick.direction": "in",
    'legend.frameon': False,
    'savefig.dpi': 600,
    'figure.dpi' : 300,
    
    'text.color': _new_black,
    'patch.edgecolor': _new_black,
    'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
    'hatch.color': _new_black,
    'axes.edgecolor': _new_black,
    'axes.titlecolor': _new_black, # should fallback to text.color
    'axes.labelcolor': _new_black,
    'xtick.color': _new_black,
    'ytick.color': _new_black,

}
# refer to https://www.nature.com/documents/nature-extended-data.pdf



def change_to_widget():
    get_ipython().run_line_magic('matplotlib', 'widget')

def apply_rcparams():
    matplotlib.rcParams.update(params_widget)


def enable_long_output():
    css_code = """
    <style>
    .output_scroll {
        height: auto !important;
        max-height: none !important;
    }
    </style>
    """
    display(HTML(css_code))

class SmartOffsetFormatter(ticker.Formatter):
    """
    Tick formatter with tight length budget and stable scaling/offset:

    Policy
    ------
    - If scaled/offset (k != 0 or C != 0): ALL non-zero labels must have
      |value| in [0.1, 10). Negatives allowed. Zero allowed anywhere.
    - Prefer solutions that keep step/10^k near {1, 2, 2.5, 5}.
    - Prefer an offset C that is DISPLAY-GRID aligned; only *display* C as
      a rounded multiple of that grid when it is already aligned; never
      "beautify" C to a different value.
    - Plain (k=0, C=0) is used only if it satisfies length+uniqueness.
    - Stickiness: reuse previous (C, k) if still valid.
    - Degrade rule: if no strict solution exists, keep a scaled view that
      satisfies the magnitude window with the max decimals allowed by the
      length budget (uniqueness may break), instead of collapsing to plain.

    Notes
    -----
    - MAX_LEN caps label length. We compute the minimal decimals to both
      fit the budget and keep uniqueness (strict path).
    - Offset text shows "×1e{k}" and "+C". The "+C" is displayed on the
      chosen display-grid if and only if C is aligned to that grid; otherwise
      it is shown exactly (no coarse rounding that would change its value).
    - axis_type='x' prints scale and offset on separate lines.
    """

    def __init__(self, offset_xy=None, offset_coords='axes',
                 offset_ha=None, offset_va=None, axis_type='y'):
        super().__init__()
        # Budgets / knobs
        self.MAX_LEN = 4
        self.K_MAX = 12
        self.OFFSET_THRESHOLD_DECADES = 1  # display grid ~ 10^(step_oom + this)

        # Placement config for offset text
        self._offset_xy = offset_xy
        self._offset_coords = offset_coords
        self._offset_ha = offset_ha
        self._offset_va = offset_va
        self.axis_type = axis_type  # 'x' => newline between scale and offset

        # Per-draw state
        self.locs = np.array([], dtype=float)
        self.abs_step = np.inf
        self._C = 0.0
        self._k = 0
        self._decimals = 0

        # Sticky state
        self._prev_C = 0.0
        self._prev_k = 0
        self._had_prev = False

    # ---------- Offset text placement patch ----------
    def set_axis(self, axis):
        super().set_axis(axis)

        def _apply(off):
            if self._offset_xy is None:
                return
            off.set_transform(axis.axes.transAxes if self._offset_coords == 'axes'
                              else axis.axes.transData)
            off.set_position(self._offset_xy)
            if self._offset_ha is not None:
                off.set_ha(self._offset_ha)
            if self._offset_va is not None:
                off.set_va(self._offset_va)
            off.set_clip_on(False)
            off.set_visible(True)

        needs_patch = (getattr(axis, "_offpos_patched_by", None) is None or
                       getattr(axis, "_offpos_patched_by", None) is not self)
        if needs_patch and hasattr(axis, "_update_offset_text_position"):
            if not hasattr(axis, "_offpos_orig_uotp"):
                axis._offpos_orig_uotp = axis._update_offset_text_position

            def _patched_uotp(_self, *args, **kwargs):
                ret = _self._offpos_orig_uotp(*args, **kwargs)
                _apply(_self.get_offset_text())
                return ret

            axis._update_offset_text_position = types.MethodType(_patched_uotp, axis)
            axis._offpos_patched_by = self

    # ---------- Helpers ----------
    @staticmethod
    def _finite(arr):
        m = np.isfinite(arr)
        return arr[m], m

    @staticmethod
    def _finite_unique(arr):
        m = np.isfinite(arr)
        return np.unique(arr[m])

    def _robust_step(self, vals):
        u = self._finite_unique(vals)
        if u.size >= 2:
            d = np.diff(np.sort(u))
            d = d[d > 0]
            if d.size > 0:
                return float(np.median(d))
        return 0.0

    def _labels(self, vals, C, k, dec):
        y = (vals - C) / (10.0 ** k)
        m = np.isfinite(y)
        y = np.where(m & (np.abs(y) < 10.0 ** (-dec - 2)), 0.0, y)  # avoid "-0"
        out = np.array([f"{v:.{dec}f}" for v in y], dtype='U')
        has_dot = np.char.find(out, '.') >= 0
        out = np.where(has_dot, np.char.rstrip(out, '0'), out)
        out = np.where(np.char.endswith(out, '.'), np.char.rstrip(out, '.'), out)
        out = np.where(out == '-0', '0', out)
        # Safety trim (rare)
        too_long = np.char.str_len(out) > self.MAX_LEN
        if np.any(too_long):
            trimmed = out.copy()
            for i in np.where(too_long)[0]:
                s = trimmed[i]
                if '.' in s:
                    s = s.split('.', 1)[0]
                    if s == '-0':
                        s = '0'
                    trimmed[i] = s
            out = trimmed
        return out

    def _len_unique_ok(self, vals, C, k, dec):
        lab = self._labels(vals, C, k, dec)
        if np.any(np.char.str_len(lab) > self.MAX_LEN):
            return False
        vfin, m = self._finite(vals)
        return np.unique(lab[m]).size >= np.unique(vfin).size

    def _needed_decimals_for_step(self, k):
        if (self.abs_step <= 0.0) or (not np.isfinite(self.abs_step)):
            return 0
        step_scaled = self.abs_step / (10.0 ** k)
        if step_scaled >= 1.0:
            return 0
        return int(np.ceil(-np.log10(step_scaled)))

    def _decimals_budget(self, vals, C, k):
        y = (vals - C) / (10.0 ** k)
        vfin, _ = self._finite(y)
        if vfin.size == 0:
            return 0, False
        has_neg = np.any(vfin < 0)
        sign_chars = 1 if has_neg else 0
        a = np.abs(vfin)
        nz = a[a > 0]
        if nz.size == 0:
            int_max = 1
        else:
            p = np.floor(np.log10(nz))
            int_max = int(np.max(p)) + 1
            int_max = max(1, int_max)
        room = self.MAX_LEN - sign_chars - int_max
        if room < 0:
            return None, has_neg
        dec_allowed = max(0, room - 1) if room >= 1 else 0  # 1 char for '.'
        return dec_allowed, has_neg

    def _pick_decimals(self, vals, C, k):
        dec_allowed, has_neg = self._decimals_budget(vals, C, k)
        if dec_allowed is None:
            return None, has_neg
        dec_need = max(0, min(dec_allowed, self._needed_decimals_for_step(k)))
        for dec in range(dec_need, dec_allowed + 1):
            if self._len_unique_ok(vals, C, k, dec):
                return dec, has_neg
        return None, has_neg

    def _max_decimals(self, vals, C, k):
        dec_allowed, has_neg = self._decimals_budget(vals, C, k)
        if dec_allowed is None:
            return 0, has_neg
        dec_need = max(0, min(dec_allowed, self._needed_decimals_for_step(k)))
        return min(dec_allowed, max(dec_need, 0)), has_neg

    def _scaled_in_target(self, vals, C, k):
        y = (vals - C) / (10.0 ** k)
        vfin, _ = self._finite(y)
        if vfin.size == 0:
            return True
        a = np.abs(vfin)
        nz = a > 0
        if not np.any(nz):
            return True
        return bool(np.all((a[nz] >= 0.1) & (a[nz] < 10.0)))

    def _good_k_core(self, step):
        # Bring step/10^k into [1,10), then bias toward {1,2,2.5,5}.
        if step <= 0 or not np.isfinite(step):
            return 0
        k = int(np.floor(np.log10(step)))  # step/10^k in [1,10)
        return k

    @staticmethod
    def _nice_score(step_scaled):
        # Prefer closeness to {1,2,2.5,5}
        nice = np.array([1.0, 2.0, 2.5, 5.0])
        return float(np.min(np.abs(np.log(step_scaled / nice))))

    def _display_grid(self, step):
        if step > 0 and np.isfinite(step):
            step_oom = int(np.floor(np.log10(step)))
        else:
            step_oom = 0
        gexp = step_oom + self.OFFSET_THRESHOLD_DECADES
        return 10.0 ** gexp  # display grid size

    @staticmethod
    def _aligned(value, grid, tol=5e-12):
        if grid == 0 or not np.isfinite(grid):
            return False
        # Alignment test: distance to nearest grid multiple is tiny
        r = np.round(value / grid)
        return bool(np.abs(value - r * grid) <= tol * max(1.0, np.abs(value)))

    # ---------- Core selection ----------
    def set_locs(self, locs):
        self.locs = np.asarray(locs, dtype=float)
        if self.locs.size == 0:
            self.abs_step = np.inf
            self._C = 0.0; self._k = 0; self._decimals = 0
            return

        self.abs_step = self._robust_step(self.locs)
        grid_disp = self._display_grid(self.abs_step)

        # Candidate offsets:
        vfin, _ = self._finite(self.locs)
        med = float(np.nanmedian(vfin)) if vfin.size else 0.0
        first = float(vfin[0]) if vfin.size else 0.0

        def snap(v):
            if not np.isfinite(v):
                return 0.0
            return float(np.round(v / grid_disp) * grid_disp)

        C_candidates = []
        if self._had_prev:
            C_candidates.append(self._prev_C)
        # prefer snapped median, then snapped first, then exact median, then 0
        C_candidates += [snap(med), snap(first), med, 0.0]

        # Candidate k around a good core guess
        k0 = self._good_k_core(self.abs_step)
        def k_order(center):
            # center first, then spread
            order = []
            for d in range(0, self.K_MAX + 1):
                for s in ([+1] if d == 0 else [-1, +1]):
                    kk = center + s * d
                    if -self.K_MAX <= kk <= self.K_MAX:
                        order.append(kk)
            return order

        # ---- A) Reuse previous if strict-valid ----
        if self._had_prev:
            dec_prev, _ = self._pick_decimals(self.locs, self._prev_C, self._prev_k)
            if dec_prev is not None:
                prev_scaled = (self._prev_k != 0) or (self._prev_C != 0.0)
                if (not prev_scaled) or self._scaled_in_target(self.locs, self._prev_C, self._prev_k):
                    self._C, self._k, self._decimals = self._prev_C, self._prev_k, dec_prev
                    return

        # ---- B) Try plain strictly ----
        dec_plain, _ = self._pick_decimals(self.locs, 0.0, 0)
        if dec_plain is not None:
            self._C, self._k, self._decimals = 0.0, 0, dec_plain
            self._prev_C, self._prev_k, self._had_prev = self._C, self._k, True
            return

        # ---- C) Search strict scaled solutions; record best degrade ----
        best_strict = None   # (score_tuple, C, k, dec)
        best_degrade = None  # (score_tuple, C, k, dec)

        for C in C_candidates:
            # choose center: prefer making step/10^k in [1,10)
            center = k0
            for k in k_order(center):
                # must satisfy magnitude window for scaled
                if not self._scaled_in_target(self.locs, C, k):
                    continue

                step_scaled = (self.abs_step / (10.0 ** k)) if (self.abs_step > 0 and np.isfinite(self.abs_step)) else 1.0
                nice_score = self._nice_score(step_scaled)
                C_aligned = self._aligned(C, grid_disp)

                # strict path
                dec_strict, _ = self._pick_decimals(self.locs, C, k)
                if dec_strict is not None:
                    score = (nice_score, int(not C_aligned), abs(k), dec_strict)
                    cand = (score, C, k, dec_strict)
                    if (best_strict is None) or (cand[0] < best_strict[0]):
                        best_strict = cand
                    # keep looking; we want the best strict

                # degrade candidate (ignore uniqueness, keep budget)
                dec_max, _ = self._max_decimals(self.locs, C, k)
                score_d = (nice_score, int(not C_aligned), abs(k), -dec_max)
                cand_d = (score_d, C, k, dec_max)
                if (best_degrade is None) or (cand_d[0] < best_degrade[0]):
                    best_degrade = cand_d

        # Take the best strict if exists
        if best_strict is not None:
            _, C, k, dec = best_strict
            self._C, self._k, self._decimals = C, k, dec
            self._prev_C, self._prev_k, self._had_prev = self._C, self._k, True
            return

        # ---- D) Degrade to the best scaled view ----
        if best_degrade is not None:
            _, C, k, dec = best_degrade
            self._C, self._k, self._decimals = C, k, dec
            self._prev_C, self._prev_k, self._had_prev = self._C, self._k, True
            return

        # ---- E) Last resort: plain best-effort ----
        dec_allowed, _ = self._decimals_budget(self.locs, 0.0, 0)
        if dec_allowed is None:
            dec_allowed = 0
        dec_need = self._needed_decimals_for_step(0)
        self._C, self._k, self._decimals = 0.0, 0, min(dec_allowed, max(0, dec_need))
        self._prev_C, self._prev_k, self._had_prev = self._C, self._k, True

    # ---------- Render a single tick ----------
    def __call__(self, x, pos=None):
        y = (x - self._C) / (10.0 ** self._k)
        if np.isfinite(y) and np.isclose(y, 0.0):
            y = 0.0
        s = f"{y:.{self._decimals}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        if s == '-0':
            s = '0'
        if len(s) > self.MAX_LEN and '.' in s:
            s = s.split('.', 1)[0]
            if s == '-0':
                s = '0'
        return s

    # ---------- Offset text (grid-aware but faithful to C) ----------
    def get_offset(self):
        parts = []
        if self._k != 0:
            parts.append(f'×1e{self._k}')
        if self._C != 0.0:
            # Display-round C only if it is ALREADY aligned to the display grid.
            grid_disp = self._display_grid(self.abs_step)
            if self._aligned(self._C, grid_disp):
                # decimals implied by grid
                if grid_disp > 0 and np.isfinite(grid_disp):
                    dec = max(0, -int(np.floor(np.log10(grid_disp))))
                else:
                    dec = 0
                s = f"{np.round(self._C, dec):.{dec}f}"
                if '.' in s:
                    s = s.rstrip('0').rstrip('.')
            else:
                # Not aligned: show exact (compact) numeric string, no coarse rounding
                s = np.format_float_positional(self._C, unique=True, trim='-')
            if s == '-0':
                s = '0'
            if s and s[0] not in '+-':
                s = '+' + s
            parts.append(s)
        return '\n'.join(parts) if self.axis_type == 'x' else ''.join(parts)


_DISPLAY_HANDLES = {}
_display_id = 0

def display_immediately(fig):
    global _display_id
    canvas = fig.canvas
    handle = display(canvas, display_id=f'{_display_id}')
    _display_id += 1


    canvas._handle_message(canvas, {'type': 'send_image_mode'}, [])
    canvas._handle_message(canvas, {'type': 'refresh'}, [])
    canvas._handle_message(canvas, {'type': 'initialized'}, [])
    canvas._handle_message(canvas, {'type': 'draw'}, [])
    
    _DISPLAY_HANDLES[fig] = handle
    return handle

def save_and_close_previous():
    active_figs = plt.get_fignums()           # [1, 2, ...]
    for num in active_figs:
        fig = plt.figure(num)               
        handle = _DISPLAY_HANDLES.get(fig)
        if handle is not None:
            handle.update(fig)
    plt.close('all')
    _DISPLAY_HANDLES.clear()

    
        
class BaseLivePlotter(ABC):
    """
    if not isinstance(data_x, np.ndarray):
        raise TypeError(f"data_x must be a numpy.ndarray, got {type(data_x)}")
    if not isinstance(data_y, np.ndarray):
        raise TypeError(f"data_y must be a numpy.ndarray, got {type(data_y)}")

    expected_y_ndim = 2
    if data_y.ndim != expected_y_ndim:
        raise ValueError(
            f"data_y must have {expected_y_ndim} dims (data_x.ndim+1), "
            f"got ndim={data_y.ndim}"
        )

    if len(data_y) != len(data_x):
        raise ValueError(
            f"data_y and data_x must have same length"
        )

    data_x has shape n, e.g. np.array([1,2,3]) or np.array([(1,1), (1,2), (1,3)]), (x, y) form
    data_y has shape n*k e.g. 
    """

    def __init__(self, data_x=np.array([[i,] for i in np.arange(100)]), data_y=np.array([[i,] for i in np.arange(100)]), labels=['X', 'Y', 'Z'], 
        update_time=0.1, fig=None, relim_mode='normal'):

        self.labels = labels
        self.xlabel = labels[0]
        self.ylabel = labels[1]
        self.zlabel = labels[-1]

        if not isinstance(data_x, np.ndarray):
            raise TypeError(f"data_x must be a numpy.ndarray, got {type(data_x)}")
        if not isinstance(data_y, np.ndarray):
            raise TypeError(f"data_y must be a numpy.ndarray, got {type(data_y)}")

        expected_y_ndim = 2
        if data_y.ndim != expected_y_ndim:
            raise ValueError(
                f"data_y must have {expected_y_ndim} dims, "
                f"got ndim={data_y.ndim}"
            )

        if len(data_y) != len(data_x):
            raise ValueError(
                f"data_y and data_x must have same length"
            )


        self.data_x = data_x
        self.data_y = data_y
        self.points_total = len(self.data_x)
        self.points_done = 0
        # used for non D1Live case, np.hist needs all number input
        self.update_time = update_time
        self.ylim_max = 100
        self.ylim_min = 0
        self.fig = fig
        if fig is None:
            self.have_init_fig = False
        else:
            self.have_init_fig = True

        self.repeat_cur = 1
        self.repeat_label = self.repeat_cur
        # assign value by self.choose_selector()
        self.relim_mode = relim_mode
        self.valid_relim_mode = ['normal', 'tight']

        self.blit_axes = []
        self.blit_artists = []
        # array contains axe and artist to be updated using blit
        self.line_colors = ['grey', 'skyblue']

        self.fixed_data_px = (480, 360)
        # canvas area (700, 500)
        # left 220, 140
        self.margins_px    = (110, 110, 100, 40)
        # (L, R, B, T)


    def create_axes_fixed(self, data_px, margins_px):
        """
        Always build a fixed-size data area using pixels + point margins.
        No fallback. Must be called with both arguments.

        Parameters
        ----------
        data_px : tuple(int, int)
            (width_px, height_px) of the data box (axes area), in pixels.
        margins_px : tuple(float, float, float, float)
            (L, R, B, T) margins around the data box, in pixels

        Returns
        -------
        matplotlib.axes.Axes
            The main axes occupying the fixed data box.
        """
        assert isinstance(data_px, (list, tuple)) and len(data_px) == 2
        assert isinstance(margins_px, (list, tuple)) and len(margins_px) == 4

        dpi = self.fig.dpi
        w_in = data_px[0] / dpi
        h_in = data_px[1] / dpi
        L, R, B, T = [m/dpi for m in margins_px]  # px -> inch

        # Store for later use by layout_split
        self._fixed_box_in = (w_in, h_in)

        # Figure size = margins + data box
        figW = L + w_in + R
        figH = B + h_in + T
        self.fig.set_size_inches(figW, figH, forward=True)

        # Create a placeholder axes and relocate it via Divider
        ax = self.fig.add_axes([0, 0, 1, 1])
        divider = Divider(
            self.fig,
            (0, 0, 1, 1),
            horizontal=[Size.Fixed(L), Size.Fixed(w_in), Size.Fixed(R)],
            vertical=[Size.Fixed(B), Size.Fixed(h_in), Size.Fixed(T)]
        )
        # Remember normalized bounds so additional columns align perfectly
        self._fixed_bounds_frac = (L/figW, B/figH, w_in/figW, h_in/figH)

        # Put the axes into the data cell (nx=1, ny=1)
        ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        return ax

    def layout_split(self, widths_rel, pads_rel):
        """
        Split the fixed data box horizontally into N columns.
        This replaces _layout_cols/layout_main_* helpers.

        Parameters
        ----------
        widths_rel : list[float]
            Relative widths of each content column measured against the
            data-box width. Sum(widths_rel) + Sum(pads_rel) <= 1.0
        pads_rel : list[float]
            Relative gaps BETWEEN columns (len = len(widths_rel) - 1).

        Returns
        -------
        list[matplotlib.axes.Axes]
            A list of axes for each column. The first element reuses self.axes.
        """
        assert len(widths_rel) >= 1
        assert len(pads_rel) == len(widths_rel) - 1, "pads_rel must be between columns."

        w_in, h_in = self._fixed_box_in  # set by create_axes_fixed
        # Build [col0, pad0, col1, pad1, ...] in absolute inches
        horiz = []
        for i, wrel in enumerate(widths_rel):
            horiz.append(Size.Fixed(wrel * w_in))
            if i < len(pads_rel):
                horiz.append(Size.Fixed(pads_rel[i] * w_in))
        vert = [Size.Fixed(h_in)]

        bounds = self._fixed_bounds_frac  # (x0, y0, w, h) in figure-fraction
        subdiv = Divider(self.fig, bounds, horizontal=horiz, vertical=vert)

        # Reuse the current self.axes for first column
        self.axes.set_axes_locator(subdiv.new_locator(nx=0, ny=0))
        axes_list = [self.axes]

        # Create extra axes for subsequent columns at nx = 2, 4, 6, ...
        for i in range(1, len(widths_rel)):
            nx = 2 * i
            ax = self.fig.add_axes([0, 0, 1, 1])
            ax.set_axes_locator(subdiv.new_locator(nx=nx, ny=0))
            axes_list.append(ax)
        return axes_list
        
    def init_figure_and_data(self):
        apply_rcparams()
        # apply the style sheet
        if not self.have_init_fig:
            with plt.ioff():
                # avoid double display from display_immediately
                self.fig = plt.figure()

            self.fig.canvas.toolbar_visible = False
            self.fig.canvas.header_visible = False
            self.fig.canvas.footer_visible = False
            self.fig.canvas.resizable = False
            self.fig.canvas.capture_scroll = True
            display_immediately(self.fig)
            self.fig.canvas.layout.display = 'none'
            # set to invisble to skip the inti fig display
            self.clear_all() # make sure no residual artist
            self.axes = self.create_axes_fixed(self.fixed_data_px, self.margins_px)
        else:
            for ax in self.fig.axes[:]:
                self.fig.delaxes(ax)
            self.fig.clear()
            self.clear_all() # make sure no residual artist
            self.fig.canvas.draw()
            # update canvas so no residual from previous plot
            self.axes = self.create_axes_fixed(self.fixed_data_px, self.margins_px)
            
        self.axes_formatter = SmartOffsetFormatter(offset_xy=(0, 1.005), offset_ha='left', offset_va='bottom')
        self.axes.yaxis.set_major_formatter(self.axes_formatter)
        # make sure no long ticks induce cut off of label

        self.axes.xaxis.set_major_formatter(SmartOffsetFormatter(offset_xy=(0.9, -0.1), offset_ha='left', offset_va='top', axis_type='x'))
        # make sure no long ticks induce cut off of label


        self.init_core() 

        if not self.have_init_fig:      
            self.fig.canvas.layout.display = 'initial'
            # set fit to visible 

        self.fig.canvas.draw()
        
        for ax in self.fig.axes:
            ticks = (ax.xaxis.get_ticklines()
                   + ax.yaxis.get_ticklines()
                   + ax.xaxis.get_minorticklines()
                   + ax.yaxis.get_minorticklines())
            for line in ticks:
                line.set_animated(True)
                line.set_clip_box(ax.bbox)
                self.blit_axes.append(ax)
                self.blit_artists.append(line)
            for spine in ax.spines.values():
                spine.set_animated(True)
                self.blit_axes.append(ax)
                self.blit_artists.append(spine)
        # in order to eliminate the blinking of ticks, due to imshow/lines in blit draw on top of spines/ticks
        
        self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # store bg


    def _start(self):
        # example of simple function
        # normally need a controller to handle
        # handle how to start liveplotter loop based on mode
        # because plot has to be the main loop so do exception check
        # and also stop all other subthreads/subfunctions

        self.init_figure_and_data()
        # initiate liveplotter and display right away with a blank fig
        try:
            while True:
                time.sleep(self.update_time)
                self.update_figure()
        except BaseException as e:
            print(e)
            self.after_plot()
            return self.fig

        # dead loop can not be keyboard interupt
    
        
    def update_figure(self, repeat_cur=None, points_done=None):
        # user's resonpibility to make sure no update_figure when data_y is all np.nan 
        # points_done needed if use np.histogram
        if repeat_cur is None:
            self.repeat_cur = 1
        else:
            self.repeat_cur = repeat_cur
        if points_done is None:
            self.points_done = self.points_total
        else:
            self.points_done = points_done

        self.update_core()

        for axe, artist in zip(self.blit_axes, self.blit_artists):
            axe.draw_artist(artist)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        
    @abstractmethod    
    def init_core(self):
        pass
    @abstractmethod     
    def update_core(self):
        pass
    @abstractmethod     
    def choose_selector(self):
        pass
    @abstractmethod 
    def set_ylim(self):
        pass


    def after_plot(self):

        for axe, artist in zip(self.blit_axes, self.blit_artists):
            artist.set_animated(False)

        self.axes.set_autoscale_on(False)
        self.fig._live_plotter = self
        # Note: Attach live_plot and controller references to fig explicitly to ensure stable lifetime management.
        # This prevents unexpected garbage collection of live_plot, controller, 
        # and selectors, maintaining robust interactive behavior.
        # because fig is always referred by display_immediately or pyqtgui so a solid base and will not be GC
        self.choose_selector()



            
    def clear_all(self):
        # code to remove selectors' plot
        if getattr(self.fig, '_live_plotter', None) is not None:
            if getattr(self.fig._live_plotter, 'area', None) is not None:
                self.fig._live_plotter.area.destroy()
                self.fig._live_plotter.area = None
            if getattr(self.fig._live_plotter, 'cross', None) is not None:
                self.fig._live_plotter.cross.destroy()
                self.fig._live_plotter.cross = None
            if getattr(self.fig._live_plotter, 'zoom', None) is not None:
                self.fig._live_plotter.zoom.destroy()
                self.fig._live_plotter.zoom = None
            if getattr(self.fig._live_plotter, 'drag', None) is not None:
                self.fig._live_plotter.drag.destroy()
                self.fig._live_plotter.drag = None
        self.fig._live_plotter = None
        
        for ax in self.fig.axes:
            for line in list(ax.lines):
                line.remove()
            for coll in list(ax.collections):
                coll.remove()
            for im in list(ax.images):
                im.remove()
            for txt in list(ax.texts):
                txt.remove()
            for patch in list(ax.patches):
                if patch not in ax.spines.values():
                    patch.remove()
            if ax.get_legend() is not None:
                ax.get_legend().remove()


    def update_verts(self, bins, counts, verts, mode='horizontal'):
        if mode=='horizontal':
            left = bins[:-1]
            right = bins[1:]
            counts = counts
            verts[:, 0, 0] = 0
            verts[:, 0, 1] = left
            verts[:, 1, 0] = counts
            verts[:, 1, 1] = left
            verts[:, 2, 0] = counts
            verts[:, 2, 1] = right
            verts[:, 3, 0] = 0
            verts[:, 3, 1] = right
        elif mode=='vertical':
            left = bins[:-1]
            right = bins[1:]
            counts = counts
            verts[:, 0, 0] = left
            verts[:, 0, 1] = 0
            verts[:, 1, 0] = left
            verts[:, 1, 1] = counts
            verts[:, 2, 0] = right
            verts[:, 2, 1] = counts
            verts[:, 3, 0] = right
            verts[:, 3, 1] = 0

    def fill_grid(self):
        grid = np.full(self.data_shape, np.nan)
        for (x, y), z in zip(self.data_x, self.data_y[:,0]):
            ix = np.searchsorted(self.x_array, x)
            iy = np.searchsorted(self.y_array, y)
            grid[iy, ix] = z
        return grid

    def _generate_square_from_extent(self, ext):
        l, r, b, u = ext

        w = r - l
        h = b - u
        if w >= h:
            pad = (w - h)/2
            b += pad
            u -= pad
        else:
            pad = (h - w)/2
            l -= pad
            r += pad
        return [l, r, b, u]



    def relim(self):
        # return 1 if need redraw
        # accept relim mode 'tight' or 'normal'
        # 'tight' will relim to fit upper and lower bound
        # 'normal' will relim to fit 0 and upper bound
        
        max_data_y = np.nanmax(self.data_y[:,0])
        min_data_y = np.nanmin(self.data_y[:,0])


        if min_data_y < 0:
            self.relim_mode = 'tight'
            # change relim mode if not able to keep 'normal' relim

        if self.relim_mode == 'normal':
            data_range = max_data_y - 0
        elif self.relim_mode == 'tight':
            data_range = max_data_y - min_data_y

        if self.relim_mode == 'normal':

            if 0<=(self.ylim_max-max_data_y)<=0.3*data_range:
                return False

            self.ylim_min = 0
            self.ylim_max = max_data_y*1.2


            self.set_ylim()
            return True

        elif self.relim_mode == 'tight':

            if 0<=(self.ylim_max - max_data_y)<=0.2*data_range and 0<=(min_data_y - self.ylim_min)<=0.2*data_range:
                return False

            self.ylim_min = min_data_y - 0.1*data_range
            self.ylim_max = max_data_y + 0.1*data_range

            if self.ylim_min!=self.ylim_max:
                self.set_ylim()
            return True

           
            
class Live1D(BaseLivePlotter):
    
    def init_core(self):
        self.lines = self.axes.plot(self.data_x[:,0], self.data_y, animated=True, alpha=1)
        for i, line in enumerate(self.lines):
            line.set_color(self.line_colors[i % len(self.line_colors)])
            self.blit_axes.append(self.axes)
            self.blit_artists.append(line)

        self.axes.set_xlim(self.data_x[0,0], self.data_x[-1,0]) # use index not min/max otherwise different orders under different units
        self.axes.set_ylim(self.ylim_min, self.ylim_max)

        self.ylabel = self.ylabel + 'x1'
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlabel(self.xlabel)
        
    def update_core(self):
        
        if (self.repeat_label!=self.repeat_cur):
            self.ylabel = self.labels[1] + f' x{self.repeat_cur}'
            self.repeat_label = self.repeat_cur
            self.axes.set_ylabel(self.ylabel)

            self.relim()
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        else:
            lim_changed = self.relim()
            if lim_changed:
                self.fig.canvas.draw()
                self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

            
        self.fig.canvas.restore_region(self.bg_fig)
        for i, line in enumerate(self.lines):
            line.set_data(self.data_x[:,0], self.data_y[:, i])

    def set_ylim(self):
        self.axes.set_ylim(self.ylim_min, self.ylim_max)
        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])


class LiveLiveDis(BaseLivePlotter):
    # live_plot class for realizing live plot plus distribution of counts
    # default live_plot class for live() function
    
    def init_core(self):

        _, self.axdis = self.layout_split([0.825, 0.15], [0.025])
        # array of plot areas in ratios and array of pads in ratios
        self.axdis.sharey(self.axes) 

        warnings.filterwarnings("ignore", category=OptimizeWarning)
        self.update_time_meter = 0.2
        # update data meter every update_time_meter second

        self.lines = self.axes.plot(self.data_x[:,0], self.data_y, animated=True, alpha=1)
        for i, line in enumerate(self.lines):
            line.set_color(self.line_colors[i % len(self.line_colors)])
            self.blit_axes.append(self.axes)
            self.blit_artists.append(line)

        self.axes.set_xlim(np.nanmin(self.data_x[:,0]), np.nanmax(self.data_x[:,0]))
        self.axes.set_ylim(self.ylim_min, self.ylim_max)

        self.ylabel = self.ylabel + 'x1'
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlabel(self.xlabel)

        self.axdis.xaxis.set_major_locator(AutoLocator())
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.relim()
        self.axdis.autoscale_view()
        self.axdis.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        self.axdis.tick_params(axis='both', which='both',bottom=False,top=False)
        # reset axdis ticks, labels

        self.counts_max = 10
        # filter out zero data
        self.n_bins = np.max((3, np.min((self.points_total//4, 50))))
        self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                        bins=self.n_bins, range=(self.ylim_min, self.ylim_max))

        self.verts = np.empty((self.n_bins, 4, 2))
        self.update_verts(self.bins, self.n, self.verts)
        self.poly = matplotlib.collections.PolyCollection(self.verts, facecolors='grey', animated=True)
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, self.counts_max)
        self.blit_axes.append(self.axdis)
        self.blit_artists.append(self.poly)
        # use collection to manage hist patches



        self.poisson_fit_line, = self.axdis.plot(self.data_y[:, 0], [0 for data in self.data_y], color='orange', animated=True, alpha=1)
        self.blit_axes.append(self.axdis)
        self.blit_artists.append(self.poisson_fit_line)

        self.points_done_fits = self.points_done
        self.ylim_min_dis = self.ylim_min
        self.ylim_max_dis = self.ylim_max

        self.last_data_time = time.time()


    @staticmethod
    def _gauss_func(x, A, mu, sigma):
        return A * np.exp(- (x - mu)**2 / (2.0 * sigma**2))


    def update_fit(self):
        # update fitting

        if not (self.points_done - self.points_done_fits)>=10:
            return
        else:
            self.points_done_fits = self.points_done

        mask = self.n > 0
        bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        bin_centers_fit = bin_centers[mask]
        counts_fit = self.n[mask]

        
        if len(bin_centers_fit) == 0:
            popt = None
        else:
            try:
                popt, pcov = curve_fit(
                    self._gauss_func,
                    bin_centers_fit,
                    counts_fit,
                    p0=[np.max(counts_fit), np.mean(bin_centers_fit), (np.max(bin_centers_fit)-np.min(bin_centers_fit))/4],
                    bounds=([0, np.min(bin_centers_fit), (np.max(bin_centers_fit)-np.min(bin_centers_fit))/10], \
                            [np.max(counts_fit)*4, np.max(bin_centers_fit), (np.max(bin_centers_fit)-np.min(bin_centers_fit))*10])
                )
            except Exception as e:
                popt = None

        if popt is not None:
            x_fit = np.sort(np.append(np.linspace(self.ylim_min, self.ylim_max, 100), bin_centers))
            y_fit = self._gauss_func(x_fit, *popt)
            
            if hasattr(self, 'poisson_fit_line') and self.poisson_fit_line is not None:
                self.poisson_fit_line.set_data(y_fit, x_fit)

            if popt[1]<=0:
                ratio = 0
                result = f'$\\sigma$={ratio:.2f}$\\sqrt{{\\mu}}$'
            else:
                ratio = popt[2]/np.sqrt(popt[1])
                if ratio <= 0.01: # ratio<1 means not a poisson distribution
                    ratio = popt[2]/popt[1]
                    result = f'$\\sigma$={ratio:.1e}$\\mu$'
                else:
                    result = f'$\\sigma$={ratio:.2f}$\\sqrt{{\\mu}}$'

            if not hasattr(self, 'fit_text'):
                self.fit_text = self.axdis.text(0.5, 1.005, 
                                                  result, transform=self.axdis.transAxes, 
                                                  color='orange', ha='center', va='bottom', animated=True,
                                                  fontsize=matplotlib.rcParams['legend.fontsize'])
                self.blit_artists.append(self.fit_text)
                self.blit_axes.append(self.axdis)
            else:
                self.fit_text.set_text(result)

        
    def update_core(self):

        lim_changed = self.relim()
        lim_dis_changed = self.relim_dis()
        
        if lim_dis_changed or lim_changed or (self.repeat_label!=self.repeat_cur):
            self.ylabel = self.labels[1] + f' x{self.repeat_cur}'
            self.repeat_label = self.repeat_cur
            self.axes.set_ylabel(self.ylabel)
            self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.n_bins, range=(self.ylim_min_dis, self.ylim_max_dis))

            self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
            self.axdis.set_xlim(0, self.counts_max)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        else: # no need to update bins positions
            self.n, _ = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.bins)
            if np.max(self.n) > self.counts_max:
                self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
                self.axdis.set_xlim(0, self.counts_max)

                self.fig.canvas.draw()
                self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw  

        self.fig.canvas.restore_region(self.bg_fig)
        for i, line in enumerate(self.lines):
            line.set_data(self.data_x[:,0], self.data_y[:, i])

        self.update_data_meter()
        self.update_dis()
        self.update_fit()

    def update_dis(self):
        self.update_verts(self.bins, self.n, self.verts)
        self.poly.set_verts(self.verts)

    def update_data_meter(self):
        if (time.time()-self.last_data_time) < self.update_time_meter:
            return
        self.last_data_time = time.time()
        newest_data = self.data_y[0, 0]
        if np.isnan(newest_data):
            return
        # may have a condition race here that measurement roll the array but not yet assign the first one to newest data
        if 1e-4<=np.abs(newest_data)<=1e4:
            oom = np.floor(np.log10(self.axes_formatter.abs_step))
            newest_data_str = f'{newest_data:.{0 if oom>=0 else -int(oom)}f}'
        else:
            newest_data_str = f'{newest_data:.1e}'
        if not hasattr(self, 'text'):
            self.text = self.axes.text(0.9, 1.005, 
                                              newest_data_str, transform=self.axes.transAxes, 
                                              color='grey', ha='right', va='bottom', animated=True, 
                                              fontsize=matplotlib.rcParams['legend.fontsize'])
            self.blit_artists.append(self.text)
            self.blit_axes.append(self.axes)
        else:
            self.text.set_text(newest_data_str)


    def relim_dis(self):
        # return 1 if need redraw, only calculate relim of main data (self.data_y[:, 0])
        max_data_y = np.nanmax(self.data_y[:, 0])
        min_data_y = np.nanmin(self.data_y[:, 0])

        data_range = max_data_y - min_data_y


        if 0<=(self.ylim_max_dis - max_data_y)<=0.2*data_range and 0<=(min_data_y - self.ylim_min_dis)<=0.2*data_range:
            return False

        self.ylim_min_dis = min_data_y - 0.1*data_range
        self.ylim_max_dis = max_data_y + 0.1*data_range
        return True



    def set_ylim(self):
        if self.ylim_min!=self.ylim_max:        
            self.axes.set_ylim(self.ylim_min, self.ylim_max)
            self.axdis.set_ylim(self.ylim_min, self.ylim_max)

        
    def choose_selector(self):
        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])

        


class Live2DDis(BaseLivePlotter):

    def init_core(self):

        _, self.axdis, self.cax = self.layout_split([0.75, 0.1, 0.1], [0.025, 0.025])
        # array of plot areas in ratios and array of pads in ratios 

        pts = self.data_x
        zvals = self.data_y[:, 0]
        x_array = np.unique(pts[:, 0])
        y_array = np.unique(pts[:, 1])
        # unique will auto sort the array
        self.x_array, self.y_array = x_array, y_array
        self.data_shape = (len(self.y_array), len(self.x_array))
        grid = self.fill_grid()
        try:
            cmap_ = matplotlib.cm.get_cmap('inferno')
        except Exception as e:
            cmap_ = plt.get_cmap('inferno')
            # in Matplotlib 3.9 or newer
        cmap = cmap_.copy()
        self.bad_color = 'white'
        cmap.set_bad(self.bad_color)
        half_step_x = 0.5*(self.x_array[-1] - self.x_array[0])/len(self.x_array)
        half_step_y = 0.5*(self.y_array[-1] - self.y_array[0])/len(self.y_array)
        extents = [self.x_array[0]-half_step_x, self.x_array[-1]+half_step_x, \
                   self.y_array[-1]+half_step_y, self.y_array[0]-half_step_y] #left, right, bottom, up


        self.lines = [self.fig.axes[0].imshow(grid, animated=True, alpha=1, cmap=cmap, extent=extents),]
        self.axes.set_anchor('W')
        self.axes.set_aspect('equal', adjustable='box')

        ext_sq = self._generate_square_from_extent(extents)
        self.axes.set_xlim(ext_sq[0], ext_sq[1])
        self.axes.set_ylim(ext_sq[2], ext_sq[3])
        self.extents_square = ext_sq
        # align to left while keep imshow ratio same as data ratio

        self.cbar = self.fig.colorbar(self.lines[0], cax = self.cax)
        #self.fig.axes[0].set_xlim((extents[0], extents[1]))
        #self.fig.axes[0].set_ylim((extents[2], extents[3]))

        self.counts_max = 10
        # filter out zero data

        self.cbar.formatter = SmartOffsetFormatter(
            offset_xy=(0.5, 1.01),
            offset_coords='axes',
            offset_ha='center',
            offset_va='bottom'
        )

        from matplotlib.ticker import MaxNLocator
        self.axdis.xaxis.set_major_locator(
            MaxNLocator(nbins=1, prune='lower')
        )

        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.relim()
        self.axdis.autoscale_view()
        # reset axdis ticks, labels
        self.n_bins = np.max((3, np.min((self.points_total//4, 50))))

        self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                        bins=self.n_bins, range=(self.ylim_min, self.ylim_max))

        self.verts = np.empty((self.n_bins, 4, 2))
        self.update_verts(self.bins, self.n, self.verts)
        self.poly = matplotlib.collections.PolyCollection(self.verts, facecolors='grey', animated=True)
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, self.counts_max)
        self.blit_axes.append(self.axdis)
        self.blit_artists.append(self.poly)

        self.blit_axes.append(self.axes)
        self.blit_artists.append(self.lines[0])

        self.zlabel = self.zlabel + 'x1'
        self.cbar.set_label(self.zlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xlabel(self.xlabel)

        self.axdis.tick_params(
            axis='x', which='both', bottom=True, top=False,
            labelbottom=True, labeltop=False
        )
        self.axdis.tick_params(
            axis='y', which='both', left=True, right=False,
            labelleft=False, labelright=False
        )

        
    def update_core(self):
        lim_changed = self.relim()
        if lim_changed or (self.repeat_label!=self.repeat_cur):

            self.zlabel = self.labels[-1] + f' x{self.repeat_cur}'
            self.repeat_label = self.repeat_cur
            self.cbar.set_label(self.zlabel)

            self.n, self.bins = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
            self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
            self.axdis.set_xlim(0, self.counts_max)
            self.fig.canvas.draw()
            self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw

        else: # no need to update bins positions
            self.n, _ = np.histogram(self.data_y[:self.points_done, 0],
                            bins=self.bins)
            if np.max(self.n) > self.counts_max:
                self.counts_max = np.max((np.max(self.n) + 5, int(np.max(self.n)*1.5)))
                self.axdis.set_xlim(0, self.counts_max)

                self.fig.canvas.draw()
                self.bg_fig = self.fig.canvas.copy_from_bbox(self.fig.bbox)  # update ylim so need redraw    


        self.fig.canvas.restore_region(self.bg_fig)
        grid = self.fill_grid()
        self.lines[0].set_array(grid)
        # other data just np.nan and controlled by set_bad
   
        self.update_dis()    

    def update_dis(self):
        self.update_verts(self.bins, self.n, self.verts)
        self.poly.set_verts(self.verts)  

    def set_ylim(self):
        self.lines[0].set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        
    def choose_selector(self):

        self.area = AreaSelector(self.fig.axes[0])
        self.cross = CrossSelector(self.fig.axes[0])
        self.zoom = ZoomPan(self.fig.axes[0])
        
        cmap = self.axes.images[0].colorbar.mappable.get_cmap()
        if self.points_done>0:
            y_min = np.nanmin(self.data_y)
            y_max = np.nanmax(self.data_y)

            self.ylim_min = y_min - 0.1*(y_max-y_min)
            self.ylim_max = y_max + 0.1*(y_max-y_min)

            self.lines[0].set_clim(vmin=self.ylim_min, vmax=self.ylim_max)
            if self.ylim_min!=self.ylim_max:
                self.axdis.set_ylim(self.ylim_min, self.ylim_max)

            self.line_min = self.axdis.axhline(y_min, color='grey', linewidth=matplotlib.rcParams['legend.fontsize']/2, alpha=0.3)
            self.line_max = self.axdis.axhline(y_max, color='grey', linewidth=matplotlib.rcParams['legend.fontsize']/2, alpha=0.3)

            self.line_l = self.axdis.axhline(self.ylim_min, color=cmap(0), linewidth=matplotlib.rcParams['legend.fontsize']/2)
            self.line_h = self.axdis.axhline(self.ylim_max, color=cmap(0.95), linewidth=matplotlib.rcParams['legend.fontsize']/2)

            self.cax.set_yticks([y_min, y_max])
            self.cax.set_yticklabels([f'{ytick:.0f}' for ytick in [y_min, y_max]])
            self.drag = DragHLine(self.line_l, self.line_h, self.update_clim, self.axdis)
        self.fig.canvas.draw()
        # must be here to display self.line_l etc. after plot done, don't know why?
    
    def update_clim(self):
        vmin = self.line_l.get_ydata()[0]
        vmax = self.line_h.get_ydata()[0]
        self.lines[0].set_clim(vmin, vmax)

                      


class AreaSelector():
    def __init__(self, ax):
        self.ax = ax
        self.text = None
        self.range = [None, None, None, None]
        self.callback = None
        artist = ax.get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            cmap = ax.images[0].colorbar.mappable.get_cmap()
            self.color = cmap(0.95)
        else:
            self.color = 'grey'

        self.selector = RectangleSelector(ax, self.onselect, interactive=True, useblit=False, button=[1], 
                                          props=dict(alpha=0.8, fill=False, 
                                                     linestyle='-', color=self.color),
                                          handle_props=dict(
                                                            marker='s',
                                                            markersize=matplotlib.rcParams['legend.fontsize']/2,
                                                            markeredgecolor=self.color,
                                                            markerfacecolor='white',
                                                            markeredgewidth=matplotlib.rcParams['lines.linewidth']/2
                                                        ),) 
        #set blit=True has weird bug, or implement RectangleSelector myself

        
    def on_callback(self):
        if self.callback is not None:
            self.callback()
        
    def onselect(self, eclick, erelease):
        x1, x2, y1, y2 = self.selector.extents
        # changed by rectangleselector
        
        if x1 == x2 or y1 == y2:
            self.range = [None, None, None, None]
            if self.text is not None:
                self.text.remove()
                self.text = None
            self.ax.figure.canvas.draw()
            self.on_callback()
            return
        
        self.range = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
        
        
        x_data = self.ax.get_xlim()
        y_data = self.ax.get_ylim()
        self.gap_x = np.abs(np.nanmax(x_data) - np.nanmin(x_data)) / 1000 if (len(x_data)>0) else 0.01
        self.gap_y = np.abs(np.nanmax(y_data) - np.nanmin(y_data)) / 1000 if (len(y_data)>0) else 0.01
        decimal_x = 0 if -int(np.ceil(np.log10(self.gap_x)))<0 else -int(np.ceil(np.log10(self.gap_x)))
        decimal_y = 0 if -int(np.ceil(np.log10(self.gap_y)))<0 else -int(np.ceil(np.log10(self.gap_y)))
        
        format_str = f'{{:.{decimal_x}f}}, {{:.{decimal_y}f}}'
        
        if self.text is None:
            self.text = self.ax.text(0.025, 0.975, f'({format_str.format(x1, y1)})\n({format_str.format(x2, y2)})',
                    transform=self.ax.transAxes,
                    color=self.color, ha = 'left', va = 'top'
                    ,fontsize=matplotlib.rcParams['legend.fontsize'])
        else:
            self.text.set_text(f'({format_str.format(x1, y1)})\n({format_str.format(x2, y2)})')

        self.ax.figure.canvas.draw()
        self.on_callback()
        
    def destroy(self):
        self.selector.set_active(False)


            


class CrossSelector():
    
    def __init__(self, ax):
        self.point = None
        self.ax = ax
        self.last_click_time = None
        self.xy = None #xy of D2
        self.callback = None
        artist = ax.get_children()[0]
        self._is_image = isinstance(artist, matplotlib.image.AxesImage)
        if self._is_image:
            cmap = ax.images[0].colorbar.mappable.get_cmap()
            self.color = cmap(0.95)
            self.color_dis = cmap(0.55)  # line on axdis

            # --- added: keep a handle to the live plotter and image flag ---
            self._live_plotter = getattr(self.ax.figure, '_live_plotter', None)
            self._axdis = getattr(self._live_plotter, 'axdis', None) if self._live_plotter is not None else None
        else:
            self.color = 'grey'

        # --------------------------------------------------------------
        
        self._dis_line = None  # horizontal line on axdis (y = z)
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)

    def on_callback(self):
        if self.callback is not None:
            self.callback()

    def on_press(self, event):
        if event.inaxes == self.ax:                
            if event.button == 3:  # mouse right key
                current_time = time.time()
                if self.last_click_time is None or (current_time - self.last_click_time) > 0.3:
                    self.last_click_time = current_time
                else:
                    self.last_click_time = None
                    self.remove_point()
                    self.ax.figure.canvas.draw()
                    return
                    
                x_data = self.ax.get_xlim()
                y_data = self.ax.get_ylim()
                self.gap_x = np.abs(np.nanmax(x_data) - np.nanmin(x_data)) / 1000 if (len(x_data)>0) else 0.01
                self.gap_y = np.abs(np.nanmax(y_data) - np.nanmin(y_data)) / 1000 if (len(y_data)>0) else 0.01
                decimal_x = 0 if -int(np.ceil(np.log10(self.gap_x)))<0 else -int(np.ceil(np.log10(self.gap_x)))
                decimal_y = 0 if -int(np.ceil(np.log10(self.gap_y)))<0 else -int(np.ceil(np.log10(self.gap_y)))
                format_str = f'{{:.{decimal_x}f}}, {{:.{decimal_y}f}}'
        
                x, y = event.xdata, event.ydata

                # --- added: compute z for 2D case without touching self.xy ---
                z_suffix = ""
                if self._is_image and (self._live_plotter is not None) \
                   and hasattr(self._live_plotter, 'x_array') and hasattr(self._live_plotter, 'y_array'):
                    try:
                        lp = self._live_plotter
                        # nearest-neighbor on the measurement grid
                        ix = int(np.argmin(np.abs(lp.x_array - x)))
                        iy = int(np.argmin(np.abs(lp.y_array - y)))
                        grid = lp.fill_grid()  # cheap on click, keeps logic simple
                        zval = grid[iy, ix]
                        if np.isfinite(zval):
                            # compact numeric string for z
                            z_suffix = f", {zval:.6g}"
                        else:
                            z_suffix = ", NaN"
                    except Exception:
                        # fail silently: still show (x, y)
                        pass
                # ------------------------------------------------------------
                self.xy = [x, y]  # keep existing behavior; DO NOT include z
                label_text = f'({format_str.format(x, y)}{z_suffix})'  # show z only in 2D

                if self.point is None:
                    self.vline = self.ax.axvline(x, color=self.color, linestyle='-', alpha=0.8)
                    self.hline = self.ax.axhline(y, color=self.color, linestyle='-', alpha=0.8)
                    self.text = self.ax.text(0.975, 0.975, label_text, ha='right', va='top', 
                                             transform=self.ax.transAxes, color=self.color, fontsize=matplotlib.rcParams['legend.fontsize'])
                    self.point, = self.ax.plot(x, y, 'o', alpha=0.8, color=self.color)
                else:
                    self.vline.set_xdata([x, x])
                    self.hline.set_ydata([y, y])
                    self.point.set_xdata([x, ])
                    self.point.set_ydata([y, ])
                    self.text.set_text(label_text)


                # draw/update/remove horizontal line on axdis (only 2D & valid z)
                if self._is_image and (self._axdis is not None):
                    if np.isfinite(zval):
                        if self._dis_line is None:
                            self._dis_line = self._axdis.axhline(
                                zval, color=self.color_dis, linewidth=matplotlib.rcParams['legend.fontsize']/4, alpha=0.3
                            )
                        else:
                            self._dis_line.set_ydata([zval, zval])
                    else:
                        if self._dis_line is not None:
                            self._dis_line.remove()
                            self._dis_line = None
                
                self.ax.figure.canvas.draw()
                self.on_callback()
    
    def remove_point(self):
        if self.point is not None:
            self.vline.remove()
            self.hline.remove()
            self.point.remove()
            self.text.remove()
            self.point = None
            self.xy = None
        if self._dis_line is not None:
            self._dis_line.remove()
            self._dis_line = None
        
    def destroy(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid_press)

        
class ZoomPan():
    def __init__(self, ax):
        self.ax = ax
        self.cid_scroll = self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
            
        self.dragging = False
        self.center_line = None
        self.callback = None
        self._live_plotter = getattr(self.ax.figure, '_live_plotter', None)

        artist = ax.get_children()[0]
        if isinstance(artist, matplotlib.image.AxesImage):
            self.image_type = '2D'
            self.ax.set_facecolor(self._live_plotter.bad_color)
            self.extents_square = self._live_plotter.extents_square

        else:
            self.image_type = '1D'

    def on_callback(self):
        if self.callback is not None:
            self.callback()


    def on_scroll(self, event):
        if event.inaxes == self.ax:

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.x_center = (xlim[0] + xlim[1])/2
            self.y_center = (ylim[0] + ylim[1])/2

            xlim_min = xlim[0]
            ylim_min = ylim[0]

            scale_factor = 1.1 if event.button == 'up' else (1/1.1)

            xlim = [scale_factor*(xlim_min - self.x_center) + self.x_center\
                    , self.x_center - scale_factor*(xlim_min - self.x_center)]
            ylim = [scale_factor*(ylim_min - self.y_center) + self.y_center\
                    , self.y_center - scale_factor*(ylim_min - self.y_center)]
            
            if self.image_type == '2D':
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            else:
                self.ax.set_xlim(xlim)
            self.ax.figure.canvas.draw()
            self.on_callback()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self._area = getattr(self, '_area', getattr(self._live_plotter, 'area', None))
        self.data_figure = getattr(self, 'data_figure', getattr(self._live_plotter, 'data_figure', None))
        if event.button == 2:
            if self.image_type == '1D':
                if event.dblclick:
                    if self._area.range[0] is not None:
                        # range of area_selector
                        new_x_min = self._area.range[0]
                        new_x_max = self._area.range[1]
                        new_xlim = (new_x_min, new_x_max) if self.data_figure.data_x[0]<=self.data_figure.data_x[-1] else (new_x_max, new_x_min)
                        self.ax.set_xlim(new_xlim[0], new_xlim[1])
                        self.ax.figure.canvas.draw()
                    else:
                        self.ax.set_xlim(self.data_figure.data_x[0], self.data_figure.data_x[-1])
                        self.ax.figure.canvas.draw()
                    self.on_callback()
                    return

                self.dragging = True
                self.press_x_pixel = event.x
                self.xlim0 = self.ax.get_xlim()
                self.center_line = self.ax.axvline(np.mean(self.ax.get_xlim()),
                                                   color='red', linestyle='--', alpha=0.3)
                self.ax.figure.canvas.draw()
            else:
                if event.dblclick:
                    if self._area.range[0] is not None:
                        # range of area_selector
                        range_array = self._area.range
                        range_array_flip = [range_array[0], range_array[1], range_array[3], range_array[2]]
                        range_array_square = self._live_plotter._generate_square_from_extent(range_array_flip)
                        self.ax.set_xlim(range_array_square[0], range_array_square[1])
                        self.ax.set_ylim(range_array_square[2], range_array_square[3])
                        self.ax.figure.canvas.draw()
                    else:
                        self.ax.set_xlim((self.extents_square[0], self.extents_square[1]))
                        self.ax.set_ylim((self.extents_square[2], self.extents_square[3]))
                        self.ax.figure.canvas.draw()

                    self.on_callback()
                    return

                self.dragging = True
                self.press_x_pixel = event.x
                self.press_y_pixel = event.y
                self.xlim0 = self.ax.get_xlim()
                self.ylim0 = self.ax.get_ylim()
                self.center_line = self.ax.scatter(np.mean(self.ax.get_xlim()), np.mean(self.ax.get_ylim()),
                                                   color='red', s=30, alpha=0.3)
                self.ax.figure.canvas.draw()

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        if self.image_type == '1D':
            dx_pixels = event.x - self.press_x_pixel
            bbox = self.ax.get_window_extent()
            pixel_width = bbox.width
            data_width = self.xlim0[1] - self.xlim0[0]
            dx_data = dx_pixels * data_width / pixel_width
            new_xlim = (self.xlim0[0] - dx_data, self.xlim0[1] - dx_data)
            self.ax.set_xlim(new_xlim)
            mid = np.mean(new_xlim)
            self.center_line.set_xdata([mid, mid])
            self.ax.figure.canvas.draw_idle()
        else:
            dx_pixels = event.x - self.press_x_pixel
            dy_pixels = event.y - self.press_y_pixel
            bbox = self.ax.get_window_extent()
            pixel_width = bbox.width
            pixel_height = bbox.height
            data_width_x = self.xlim0[1] - self.xlim0[0]
            data_width_y = self.ylim0[1] - self.ylim0[0]
            dx_data = dx_pixels * data_width_x / pixel_width
            dy_data = dy_pixels * data_width_y / pixel_height
            new_xlim = (self.xlim0[0] - dx_data, self.xlim0[1] - dx_data)
            new_ylim = (self.ylim0[0] - dy_data, self.ylim0[1] - dy_data)
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            mid_x, mid_y = np.mean(new_xlim), np.mean(new_ylim)
            self.center_line.set_offsets([[mid_x, mid_y]])
            self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        if event.button == 2 and self.dragging:
            self.dragging = False
            if self.center_line is not None:
                self.center_line.remove()
                self.center_line = None
            self.ax.figure.canvas.draw()
            self.on_callback()

    def destroy(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid_scroll)
        self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        self.ax.figure.canvas.mpl_disconnect(self.cid_release)
    

class DragHLine():
    def __init__(self, line_l, line_h, update_func, ax):
        self.line_l = line_l
        self.line_h = line_h
        self.ax = ax
        self.press = None
        self.update_func = update_func
        self.line_l.set_animated(True)
        self.line_h.set_animated(True)
        self.is_on_l = False
        self.is_on_h = False
        self.useblit = True
        self.background = None
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_draw = self.ax.figure.canvas.mpl_connect('draw_event', self.on_draw)
        self.last_update_time = time.time()

    def on_draw(self, event):
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.line_l)
        self.ax.draw_artist(self.line_h)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains_l, _ = self.line_l.contains(event)
        contains_h, _ = self.line_h.contains(event)
        if not (contains_l or contains_h):
            return
        
        if contains_l:
            self.is_on_l = True
            self.press = self.line_l.get_ydata(), event.ydata
        else:
            self.is_on_h = True
            self.press = self.line_h.get_ydata(), event.ydata

        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)

    def on_motion(self, event):
        if not self.press:
            return
        if event.inaxes != self.ax:
            return
        current_time = time.time()
        if current_time - self.last_update_time < 0.03:
            return
        ypress, ydata = self.press
        dy = event.ydata - ydata
        new_ydata = [y + dy for y in ypress]
        if self.is_on_l:
            self.line_l.set_ydata(new_ydata)
        if self.is_on_h:
            self.line_h.set_ydata(new_ydata)
        self.update_func()


        self.ax.figure.canvas.restore_region(self.background)
        for line in self.ax.lines:
            self.ax.draw_artist(line)
        self.ax.figure.canvas.blit(self.ax.bbox)


        self.last_update_time = current_time

    def on_release(self, event):
        self.press = None
        self.is_on_l = False
        self.is_on_h = False
        self.update_func()
        self.ax.figure.canvas.draw()
        
    def destroy(self):
        self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        self.ax.figure.canvas.mpl_disconnect(self.cid_release)
        self.ax.figure.canvas.mpl_disconnect(self.cid_motion)
        self.ax.figure.canvas.mpl_disconnect(self.cid_draw)
