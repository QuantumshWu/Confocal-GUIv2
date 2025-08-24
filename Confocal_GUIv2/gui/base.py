from qframelesswindow import FramelessWindow, StandardTitleBar
import sys, os, inspect, time
import numpy as np
from Confocal_GUIv2.helper import float2str_eng, float2str, FLOAT_PATTERN, COORDINATE_LIST_PATTERN, reentrancy_guard, str2python, python2str,\
log_error, FLOAT_OR_X_PATTERN, FLOAT_OR_X_PARSING_PATTERN, align_to_resolution, data_x_str_validator

from PyQt5 import QtCore, QtWidgets, uic, QtGui
from PyQt5.QtCore import Qt, QSize, pyqtProperty, QPropertyAnimation, QTimer, QPoint, QAbstractAnimation, QEventLoop, QEvent, QRectF, \
QRect, pyqtSignal, QObject, QRegularExpression, QEasingCurve
from PyQt5.QtGui import QColor, QPainter, QBrush, QFont, QPolygon, QTextOption, QRegularExpressionValidator, QValidator, QPen
from PyQt5.QtWidgets import (
    QAbstractButton,
    QWidget,
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QLineEdit,
    QComboBox,
    QLabel,
    QFileDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QDoubleSpinBox,
    QToolButton,
    QAbstractSpinBox,
    QPushButton,
    QMessageBox,
    QTextEdit,
    QPlainTextEdit,
    QStackedWidget,
    QSizePolicy,
    QApplication,
    QTabWidget,
    QTabBar,
    QStyleOptionTab,
    QStylePainter,
    QStyle,
    QHeaderView,
    QTableWidget,
    QStyleOptionComboBox,
    QTableWidgetItem,
    QGridLayout,
    QRadioButton,
)


ACCENT_COLOR = "#77AADD"  
HOVER_COLOR  = "#004578"
BG_COLOR     = "#F3F3F3"   
TEXT_COLOR   = "#323130"     
HINT_COLOR   = "#F0a150"
PLACEHOLDER  = "#A19F9D"      
RADIUS       = 4              
FONT_FAMILY  = "Segoe UI"
FONT_SIZE    = 12

# --- Fluent-style preset colors (picked for good contrast with white text) ---
FLUENT_GREEN  = "#7FC2AD"
FLUENT_RED    = "#CD7380"
FLUENT_ORANGE = "#D69A6E"
FLUENT_YELLOW = "#E5C85B"
FLUENT_GREY = "#A2A2A2"

PADDING_V = 1
PADDING_H = 1
COMBO_WIDTH = 16
STEP_WIDTH = 6
COMBO_TRI_SIZE = 4*2 # must be multiple of 4

class ToggleSwitch(QAbstractButton):
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)

        # Animation state
        self._offset = 3
        self._animating = False
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.stateChanged.connect(self._on_anim_state_changed)


        self.toggled.connect(self._start_animation)
        self.setMinimumSize(60, 30)
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))

    def sizeHint(self):
        return self.minimumSize()

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # ---- choose colors based on enabled/disabled state ----
        if self.isEnabled():
            thumb_color = "#FFFFFF"
            track_color = ACCENT_COLOR if self.isChecked() else PLACEHOLDER
        else:
            # Disabled: always muted; do not advertise "on" color
            thumb_color = "#FFFFFF"
            track_color = BG_COLOR

        text_color  = TEXT_COLOR

        # Track
        painter.setBrush(QBrush(QColor(track_color)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h/2, h/2)

        # Thumb position
        thumb_d = h - 6
        if self._animating:
            offset = self._offset
        else:
            offset = (w - thumb_d - 3) if self.isChecked() else 3

        # Thumb
        painter.setBrush(QBrush(QColor(thumb_color)))
        painter.drawEllipse(int(offset), 3, thumb_d, thumb_d)

        # Optional label
        if self.text():
            painter.setPen(QColor(text_color))
            painter.drawText(self.rect(), Qt.AlignCenter, self.text())

    def _start_animation(self, checked):
        """Animate when enabled; snap instantly when disabled or size is 0."""
        w, h = self.width(), self.height()
        thumb_d = h - 6
        end_pos = (w - thumb_d - 3) if checked else 3

        # NEW: skip animation if disabled (or geometry not ready)
        if not self.isEnabled() or w <= 0 or h <= 0:
            self._anim.stop()
            self._offset = end_pos
            self.update()
            return

        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(end_pos)
        self._anim.start()

    def _on_anim_state_changed(self, new_state, old_state):
        self._animating = (new_state == QAbstractAnimation.Running)
        if not self._animating:
            self.update()

    def mouseReleaseEvent(self, ev):
        # Default behavior already respects disabled state; keep it simple.
        if ev.button() == Qt.LeftButton and self.isEnabled():
            super().mouseReleaseEvent(ev)
        else:
            super().mouseReleaseEvent(ev)


    # ---- animated property ----
    def getOffset(self):
        return self._offset

    def setOffset(self, value):
        self._offset = value
        self.update()

    offset = pyqtProperty(float, fget=getOffset, fset=setOffset)


class TriStateToggleSwitch(QAbstractButton):
    """
    Capsule-style tri-state toggle switch with consistent styling.
    All three labels remain visible; the thumb slides to overlay the active segment.
    Click cycles through states: 0 -> 1 -> 2.
    Uses white thumb to cover label, with accent border indicating selection.
    Text always drawn on top in standard color.
    """
    def __init__(self, labels=None, parent=None):
        super().__init__(parent)
        self.labels = labels or ["Off", "Mid", "On"]
        self._state = 0
        self._offset = 0.0
        # Animation for smooth thumb movement
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(150)
        self._anim.stateChanged.connect(self._on_anim_state_changed)

        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(90, 30)
        self.clicked.connect(self._cycle_state)

    def sizeHint(self):
        """Return the recommended size (same as minimumSize)."""
        return self.minimumSize()

    def paintEvent(self, event):
        w = self.width()
        h = self.height()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Compute segment dimensions and margin
        segment_count = len(self.labels)
        segment_width = w / segment_count
        margin = 3
        thumb_height = h - 2 * margin
        thumb_width = segment_width - 2 * margin

        # 1) Draw full track with placeholder color
        painter.setBrush(QBrush(QColor(PLACEHOLDER)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, w, h, h / 2, h / 2)

        # 2) Draw the white thumb over active segment (no border)
        if self._anim.state() == QAbstractAnimation.Running:
            x = self._offset + margin
        else:
            x = self._state * segment_width + margin
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(
            int(x), margin,
            int(thumb_width), int(thumb_height),
            int(thumb_height / 2), int(thumb_height / 2)
        )

        # 3) Draw the labels on top
        painter.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        painter.setPen(QColor(TEXT_COLOR))
        for idx, label in enumerate(self.labels):
            rect = QRectF(idx * segment_width, 0, segment_width, h)
            painter.drawText(rect, Qt.AlignCenter, label)

        painter.end()

    def _cycle_state(self):
        """Cycle through the three states and animate thumb movement."""
        previous = self._state
        self._state = (self._state + 1) % len(self.labels)
        segment_width = self.width() / len(self.labels)

        self._anim.stop()
        self._anim.setStartValue(previous * segment_width)
        self._anim.setEndValue(self._state * segment_width)
        self._anim.start()

    def _on_anim_state_changed(self, new_state, old_state):
        """Ensure offset matches current state after animation completes."""
        if new_state != QAbstractAnimation.Running:
            segment_width = self.width() / len(self.labels)
            self._offset = self._state * segment_width
            self.update()

    def getOffset(self):
        """Getter for the animated offset property."""
        return self._offset

    def setOffset(self, value):
        """Setter for the animated offset property."""
        self._offset = value
        self.update()

    # Property for animation to adjust thumb position
    offset = pyqtProperty(float, fget=getOffset, fset=setOffset)

    def state(self):
        """Return the current selected index (0, 1, or 2)."""
        return self._state


class CenteredTabBar(QTabBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Ensure the widget‐level font matches your stylesheet
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))

    def paintEvent(self, event):
        painter = QStylePainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        ICON_TEXT_GAP = 4  # px between icon and text

        for idx in range(self.count()):
            # 1) Build the style option for this tab
            option = QStyleOptionTab()
            self.initStyleOption(option, idx)

            # 2) Draw the tab background and border via stylesheet rules
            painter.drawControl(self.style().CE_TabBarTabShape, option)

            # 3) Compute how much width the close-button takes
            btn_w = self.style().pixelMetric(
                QStyle.PM_TabCloseIndicatorWidth, option, self
            )
            # 4) Derive the 'available' rect by subtracting close button area
            avail = QRect(option.rect)
            # move the right edge left by btn_w (plus a tiny margin if desired)
            avail.setRight(avail.right() - btn_w - 2)

            # 5) Choose pen color: white on hover-unselected, else ButtonText role
            if (option.state & QStyle.State_MouseOver) and not (option.state & QStyle.State_Selected):
                painter.setPen(QColor('white'))
            else:
                painter.setPen(option.palette.color(option.palette.ButtonText))

            # 6) Use the widget’s font (ensures it’s Segoe UI 12pt)
            painter.setFont(self.font())

            # 7) Grab icon + text
            icon = self.tabIcon(idx).pixmap(self.iconSize())
            text = self.tabText(idx)

            # 8) Measure widths to center the block
            fm     = painter.fontMetrics()
            text_w = fm.horizontalAdvance(text)
            icon_w = icon.width() if not icon.isNull() else 0
            gap    = ICON_TEXT_GAP if icon_w else 0
            total  = icon_w + gap + text_w

            # 9) Compute the x/y so “icon+gap+text” is centered in avail
            x0     = avail.x() + (avail.width() - total)//2
            y_text = avail.y() + (avail.height() - fm.height())//2 + fm.ascent()
            y_icon = avail.y() + (avail.height() - icon.height())//2

            # 10) Draw icon then text
            if icon_w:
                painter.drawPixmap(x0, y_icon, icon)
            painter.drawText(x0 + icon_w + gap, y_text, text)

class FluentTabWidget(QTabWidget):
    """
    A QTabWidget styled to match Fluent design guidelines:
    - Rounded white pane background
    - Custom tab colors and hover/selected effects
    - Drop shadow for depth
    - Close button icon/color customization
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)

        self.setTabBar(CenteredTabBar())

        self.tabBar().setElideMode(Qt.ElideNone)
        # Let each tab size itself to its content instead of sharing available width
        self.tabBar().setExpanding(False)
        # Show scroll buttons when tabs exceed the widget width
        self.setUsesScrollButtons(True)

        # Apply stylesheet for Fluent appearance
        self.setStyleSheet(f"""
            /* Pane styling: white background and rounded corners */
            QTabWidget::pane {{
                background: white;
                margin: 0px;
                padding: 0px;
                border: none;
                border-top-right-radius: {RADIUS}px;
                border-bottom-left-radius: {RADIUS}px;
                border-bottom-right-radius: {RADIUS}px;
            }}

            QTabWidget QStackedWidget {{
                background: white;
                border: none;
                margin: 0px;
                padding: 0px;
                border-top-right-radius: {RADIUS}px;
                border-bottom-left-radius: {RADIUS}px;
                border-bottom-right-radius: {RADIUS}px;
            }}

            QTabWidget {{
                margin: 0px;
                padding: 0px;
            }}

            QTabWidget::tab-bar {{
                margin: 0px;
                padding: 0px;
            }}

            /* Tab styling: background, text color, padding, rounded top corners */
            QTabBar::tab {{
                margin: 0px;
                width: 120px;
                height: 30px;
                background: {BG_COLOR};
                color: {TEXT_COLOR};
                padding: {PADDING_V}px {PADDING_H}px;
                border-top-left-radius: {RADIUS}px;
                border-top-right-radius: {RADIUS}px;
                margin-right: 2px;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}

            /* Selected tab styling */
            QTabBar::tab:selected {{
                background: white;
                color: {TEXT_COLOR};
            }}

            /* Hover effect on unselected tabs */
            QTabBar::tab:!selected:hover {{
                background: {ACCENT_COLOR};
                color: white;
            }}

        """)

        # Add a subtle drop shadow under the tab widget for depth
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)



class FluentGroupBox(QGroupBox):
    def __init__(self, title='', parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(f"""
            QGroupBox {{
                background: white;
                border-radius: {RADIUS}px;
                margin-top: 0px;         
                padding-top: 32px;  
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                background: {BG_COLOR};
                padding: {PADDING_V}px {PADDING_H}px;
                border-radius: {RADIUS}px;
                color: {TEXT_COLOR};
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}
        """)
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(20); sh.setColor(QColor(0, 0, 0, 50)); sh.setOffset(0)
        self.setGraphicsEffect(sh)



class FluentFrame(QFrame):
    def __init__(self, parent=None, round=('NW', 'NE', 'SE', 'SW')):
        super().__init__(parent)

        s = {x.upper() for x in round}
        def on(*keys): return RADIUS if any(k in s for k in keys) else 0

        tl = on('NW')
        tr = on('NE')
        br = on('SE')
        bl = on('SW')

        self.setStyleSheet(f"""
            QFrame {{
                background: white;
                border: none;
                border-top-left-radius: {tl}px;
                border-top-right-radius: {tr}px;
                border-bottom-right-radius: {br}px;
                border-bottom-left-radius: {bl}px;
            }}
        """)
        sh = QGraphicsDropShadowEffect(self)
        sh.setBlurRadius(20); sh.setColor(QColor(0, 0, 0, 50)); sh.setOffset(0)
        self.setGraphicsEffect(sh)


class FluentStackedWidget(QStackedWidget):
    """
    A QStackedWidget styled with Fluent design: white background, rounded corners,
    and a subtle drop shadow effect.
    """
    def __init__(self, parent=None, shadow=True):
        super().__init__(parent)
        # Set a white background and rounded corners
        self.setStyleSheet(f"""
            QStackedWidget {{
                background: white;
                border-radius: {RADIUS}px;
            }}
        """)
        # Apply drop shadow
        if shadow is True:
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setColor(QColor(0, 0, 0, 50))  # semi-transparent black
            shadow.setOffset(0, 0)
            self.setGraphicsEffect(shadow)

class FluentSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOrientation(Qt.Horizontal)
        self.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height:6px; background:{BG_COLOR}; border-radius:{RADIUS}px; }}
            QSlider::handle:horizontal {{ width:16px; margin:-5px 0; border-radius:{RADIUS}px; background:{ACCENT_COLOR}; }}
        """ )

class FluentButton(QPushButton):
    """
    Fluent-styled button with color presets:
      set_color("green" | "red" | "orange" | "#RRGGBB" | QColor)
    Hover is always a darker version of the base by HOVER_DARKEN_PCT_DEFAULT.
    """

    _STYLE_TMPL = """
    QPushButton {{
      background: {bg};
      color: white;
      border: none;
      border-radius: {radius}px;
      padding: {pad_v}px {pad_h}px;
      font: {font_size}pt "{font_family}";
    }}
    QPushButton:hover {{ background: {hover}; }}
    QPushButton:disabled {{ background: {placeholder}; color: {bg_color}; }}
    """

    _HOVER_DARKEN_PCT = 184

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_bg = ACCENT_COLOR
        self._apply_style(self._current_bg)

    def set_color(self, bg):
        """
        Accepts:
          - 'green' | 'red' | 'orange'  (preset names)
          - '#RRGGBB' or '#RGB'         (hex string)
          - QColor
        """
        bg_hex = self._resolve_bg(bg)
        if bg_hex == self._current_bg:
            return
        self._current_bg = bg_hex
        self._apply_style(bg_hex)

    # ---------- internals ----------
    def _apply_style(self, bg_hex: str) -> None:
        hover_hex = QColor(bg_hex).darker(self._HOVER_DARKEN_PCT).name(QColor.HexRgb)
        self.setStyleSheet(self._STYLE_TMPL.format(
            bg=bg_hex,
            hover=hover_hex,
            placeholder=PLACEHOLDER,
            bg_color=BG_COLOR,
            radius=RADIUS,
            pad_v=PADDING_V,
            pad_h=PADDING_H,
            font_size=FONT_SIZE,
            font_family=FONT_FAMILY,
        ))

    def _resolve_bg(self, value) -> str:
        """Resolve preset name or color to '#RRGGBB'."""
        # QColor
        if isinstance(value, QColor):
            return value.name(QColor.HexRgb)
        s = str(value).strip()
        # hex like '#RGB' -> expand
        if len(s) == 4 and s[0] == '#':
            s = '#' + ''.join(ch * 2 for ch in s[1:])
        return s

class FluentLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setStyleSheet(f"""
            QLineEdit {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {RADIUS}px;
                padding: {PADDING_V}px {PADDING_H}px;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
                color: {TEXT_COLOR};
            }}
            QLineEdit:focus {{ border: 1px solid {ACCENT_COLOR}; }}
            QLineEdit:disabled {{ background: {BG_COLOR}; color: {PLACEHOLDER}; }}
        """)

        self._res_step = None
        self.editingFinished.connect(self._snap_to_resolution)

    def set_regex(self, pattern_obj):
        """
        Accept a compiled Python 're.Pattern' and convert to QRegularExpression (verbose mode).
        """
        rx = QRegularExpression(f"(?x){pattern_obj.pattern}")
        self.setValidator(QRegularExpressionValidator(rx, self))

    def set_resolution(self, step: float):
        """
        Enable snapping to the nearest multiple of 'step'.
        Only one numeric token may appear in the text by design.
        """
        self._res_step = float(step) if step else None

    def set_allow_zero(self, allow_zero: bool = False):
        self.allow_zero = allow_zero

    def _snap_to_resolution(self):
        """
        Runs after editing finished.
        Cases:
          - '±x'            -> do nothing
          - 'x ± number'    -> snap only that number substring
          - 'number ± x'    -> snap only that number substring
          - 'pure number'   -> snap the whole numeric part (keep outer whitespace)
        Rules:
          - If already aligned to 'step', keep original token unchanged.
          - When rewriting, try 'AeB' (A,B integers and B != 0); else output decimal with trailing zeros removed.
          - No blockers; setText will emit textChanged by design.
        """
        step = getattr(self, "_res_step", None)
        allow_zero = getattr(self, "allow_zero", False)
        if not step:
            return

        s = self.text()
        if not s:
            return

        new_s = align_to_resolution(value=s, resolution=step, allow_zero=allow_zero)
        if new_s != s:
            self.setText(new_s)


class FloatLineEdit(FluentLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_regex(FLOAT_PATTERN)

class FloatOrXLineEdit(FluentLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_regex(FLOAT_OR_X_PATTERN)

class CoordinateListLineEdit(FluentLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_regex(COORDINATE_LIST_PATTERN)



class DataXValidator(QValidator):
    """
    Minimal QValidator: Accepts when text parses to a numeric np.ndarray
    of shape (N, n_dim). Empty text is Intermediate to allow typing.
    """
    def __init__(self, n_dim: int, parent=None):
        super().__init__(parent)
        self.n_dim = int(n_dim)

    def validate(self, s: str, pos: int):
        # Allow empty while editing
        ok = data_x_str_validator(s, self.n_dim)
        return (
            QtGui.QValidator.Acceptable if ok else QtGui.QValidator.Intermediate,
            s,
            pos,
        )

class DataXLineEdit(FluentLineEdit):
    """
    Minimal wrapper: only installs DataXValidator(n_dim).
    No extra signals/methods/UI.
    """
    def __init__(self, n_dim: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setValidator(DataXValidator(n_dim, self))


class FluentTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet(f"""
            QTextEdit {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {RADIUS}px;
                padding: {PADDING_V}px {PADDING_H}px;     
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
                color: {TEXT_COLOR};
            }}
            QTextEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
            QTextEdit:disabled {{
                background: {BG_COLOR};
                color: {PLACEHOLDER};
            }}
        """)

class FluentPlainTextEdit(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {RADIUS}px;
                padding: {PADDING_V}px {PADDING_H}px;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
                color: {TEXT_COLOR};
            }}
            QPlainTextEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
            QPlainTextEdit:disabled {{
                background: {BG_COLOR};
                color: {PLACEHOLDER};
            }}
        """)

class FluentComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Use an internal line edit for consistent text layout with QLineEdit.
        self.setEditable(True)
        le = self.lineEdit()
        le.setReadOnly(True)
        le.setFrame(False)
        le.setCursor(Qt.ArrowCursor)
        le.setFocusPolicy(Qt.NoFocus)                        # no focus -> no caret jumps
        le.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # pass clicks to the combo
        le.setStyleSheet("background: transparent; border: none; padding: 0;")
        # Keep modest inner margins; right cutoff is enforced via geometry.
        le.setTextMargins(PADDING_H, PADDING_V, PADDING_H, PADDING_V)
        le.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Keep the visible text anchored on the left even when it changes.
        self.currentTextChanged.connect(self._reset_cursor)
        self.editTextChanged.connect(self._reset_cursor)

        # QSS: edit area geometry is handled in _layout_lineedit(); no padding-right here.
        self.setStyleSheet(f"""
            QComboBox {{
                background-color: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {RADIUS}px;
                color: {TEXT_COLOR};
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
                padding: 0px;
            }}
            QComboBox:disabled {{
                background-color: {BG_COLOR};
                border: 1px solid {PLACEHOLDER};
                color: {PLACEHOLDER};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: border;
                subcontrol-position: right;
                width: {COMBO_WIDTH}px;
                border: none;
                background-color: {ACCENT_COLOR};
                border-top-right-radius: {RADIUS}px;
                border-bottom-right-radius: {RADIUS}px;
            }}
            QComboBox::drop-down:disabled {{
                background-color: {PLACEHOLDER};
            }}
            QComboBox::drop-down:hover {{
                background-color: {HOVER_COLOR};
            }}
            QComboBox::down-arrow {{ image: none; }}  /* we paint the arrow ourselves */
            QComboBox QAbstractItemView {{
                border: 1px solid {PLACEHOLDER};
                selection-background-color: {ACCENT_COLOR};
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}
        """)

    # --- helpers --------------------------------------------------------------

    def _reset_cursor(self, *_):
        """Always keep the edit cursor at the beginning so the left side stays visible."""
        if self.isEditable() and self.lineEdit():
            self.lineEdit().setCursorPosition(0)

    def _layout_lineedit(self):
        """Resize the internal lineEdit so it ends before the right drop-down button."""
        if not self.isEditable():
            return
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)

        # Prefer the style-provided edit field; fall back to plain geometry.
        edit_rect = self.style().subControlRect(QStyle.CC_ComboBox, opt,
                                                QStyle.SC_ComboBoxEditField, self)
        if not edit_rect.isValid() or edit_rect.width() <= 0:
            edit_rect = QRect(0, 0, self.width() - COMBO_WIDTH, self.height())

        # Hard trim against the visual drop area to avoid any overlap.
        drop_left = self.width() - COMBO_WIDTH
        if edit_rect.right() >= drop_left:
            edit_rect.setRight(drop_left - 1)
        if edit_rect.left() < 0:
            edit_rect.setLeft(0)

        self.lineEdit().setGeometry(edit_rect)

    # Ensure geometry is correct throughout the widget lifecycle.
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_lineedit()

    def showEvent(self, e):
        super().showEvent(e)
        self._layout_lineedit()
        self._reset_cursor()

    # --- painting -------------------------------------------------------------

    def paintEvent(self, ev):
        """Let style/QSS paint first, then draw a centered white triangle in the drop area."""
        super().paintEvent(ev)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#FFFFFF")))

        # Center the arrow in the actual colored drop area (right COMBO_WIDTH px).
        drop_rect = QRectF(self.width() - COMBO_WIDTH, 0, COMBO_WIDTH, self.height())
        cx = int(drop_rect.center().x())
        cy = int(drop_rect.center().y())

        # Keep your original triangle size/proportions.
        size = COMBO_TRI_SIZE            # multiple of 4 as you defined
        half_w = size // 2
        quarter_h = size // 4

        pts = [
            QPoint(cx - half_w, cy - quarter_h),
            QPoint(cx + half_w, cy - quarter_h),
            QPoint(cx,          cy + quarter_h),
        ]
        painter.drawPolygon(QPolygon(pts))
        painter.end()


class FluentLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_COLOR};
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}
        """)


class FluentFileDialog(QFileDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(f"""
            QFileDialog {{
                background: white;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}
        """ )

class FluentMessageDialog(QDialog):
    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent, Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.setStyleSheet("QDialog { background: white; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        lbl = FluentLabel(message, self)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok = FluentButton("OK", self)
        ok.clicked.connect(self.accept)
        btn_row.addWidget(ok)
        layout.addLayout(btn_row)

class FluentMessageBox(QMessageBox):
    @staticmethod
    def warning(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        dlg = FluentMessageDialog(title, text, parent)
        dlg.exec_()

    @staticmethod
    def information(parent, title, text, buttons=QMessageBox.Ok, defaultButton=QMessageBox.NoButton):
        dlg = FluentMessageDialog(title, text, parent)
        dlg.exec_()



class FluentInputDialog(QDialog):
    def __init__(self, prompt: str, default: float, parent=None):
        super().__init__(parent, Qt.WindowTitleHint|Qt.WindowCloseButtonHint)
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.setStyleSheet(f"""
            QDialog {{
                background: white;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12,12,12,12)

        lbl = FluentLabel(prompt, self)
        layout.addWidget(lbl)

        self._edit = FluentLineEdit(self)
        self._edit.setText(str(default))
        layout.addWidget(self._edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = FluentButton("Cancel", self)
        cancel.clicked.connect(self.reject)
        ok     = FluentButton("OK", self)
        ok.clicked.connect(self.accept)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)

    def getValue(self):
        if self.exec_() == QDialog.Accepted:
            try:
                return float(self._edit.text()), True
            except ValueError:
                return None, False
        return None, False



class FluentDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, length=5, allow_minus=False, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QAbstractSpinBox.PlusMinus)
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {RADIUS}px;
                padding: {PADDING_V}px {PADDING_H}px;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
                color: {TEXT_COLOR};
            }}
            QDoubleSpinBox:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
            QDoubleSpinBox:disabled {{
                background: {BG_COLOR};
                color: {PLACEHOLDER};
            }}

            QDoubleSpinBox::up-button,
            QDoubleSpinBox::down-button {{
                subcontrol-origin: border;
                width: {COMBO_WIDTH}px;
                border: none;
                background-color: {ACCENT_COLOR};
            }}
            QDoubleSpinBox::up-button {{
                subcontrol-position: top right;
                border-top-right-radius: {RADIUS}px;
            }}
            QDoubleSpinBox::down-button {{
                subcontrol-position: bottom right;
                border-bottom-right-radius: {RADIUS}px;
            }}
            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover {{
                background-color: {HOVER_COLOR};
            }}

            QDoubleSpinBox::up-arrow,
            QDoubleSpinBox::down-arrow {{
                image: none;
            }}
        """)

        self._step_btn = QToolButton(self)
        self._step_btn.setText(".")
        self._step_btn.setCursor(Qt.PointingHandCursor)
        self._step_btn.setStyleSheet(f"""
            QToolButton {{
                background: {ACCENT_COLOR};
                color: white;
                border: none;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}
            QToolButton:hover {{
                background: {HOVER_COLOR};
            }}
        """)
        self._step_btn.clicked.connect(self._on_edit_step)


        # enforce minimum 5 length so we always have at least “0.001”
        self.length = max(length, 5)
        # set the numeric range so that integer part never exceeds length
        if allow_minus is True:
            self.setRange(1 - 10**(self.length-1), 10**self.length - 1)
        else:
            self.setRange(10**-(self.length-2), 10**self.length - 1)
        # stepping in units of 1
        self.setSingleStep(1)
        # internal precision (doesn't affect what we display)
        self.setDecimals(self.length - 2)
        # prevent the user from typing more characters than we’ll ever display
        self.lineEdit().setMaxLength(self.length)


    def setValue(self, value: float) -> None:
        rounded = eval(float2str(value, length=self.length))
        super().setValue(rounded)

    def setSingleStep(self, step: float) -> None:
        rounded = eval(float2str(step, length=self.length))
        super().setSingleStep(rounded)

    def stepBy(self, steps: int) -> None:
        current = self.value()
        step = self.singleStep()
        target = eval(float2str(current + steps * step, length=self.length))
        if self.minimum() <= target <= self.maximum():
            super(FluentDoubleSpinBox, self).setValue(target)
        # disable change less than singstep

    def textFromValue(self, value: float) -> str:
        return float2str(value, length=self.length)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        btn_w = COMBO_WIDTH
        step_sz = STEP_WIDTH         
        x = self.width() - btn_w - step_sz
        y = self.height()
        self._step_btn.setGeometry(x, 0, step_sz, y)

    def _on_edit_step(self):
        current = self.singleStep()
        dlg = FluentInputDialog("Edit step", current, self)
        dlg._edit.setMaxLength(self.lineEdit().maxLength())
        val, ok = dlg.getValue()
        if ok and val is not None:
            self.setSingleStep(val)


    def paintEvent(self, ev):
        super().paintEvent(ev)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        size  = COMBO_TRI_SIZE
        btn_w = COMBO_WIDTH
        cx    = self.width() - btn_w/2
        h     = self.height()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#FFFFFF")))

        y_up = h * 0.25
        up_pts = [
            QPoint(int(cx - size/2), int(y_up + size/4)),
            QPoint(int(cx + size/2), int(y_up + size/4)),
            QPoint(int(cx),         int(y_up - size/4)),
        ]
        painter.drawPolygon(QPolygon(up_pts))

        y_dn = h * 0.75
        dn_pts = [
            QPoint(int(cx - size/2), int(y_dn - size/4)),
            QPoint(int(cx + size/2), int(y_dn - size/4)),
            QPoint(int(cx),         int(y_dn + size/4)),
        ]
        painter.drawPolygon(QPolygon(dn_pts))
        painter.end()



class FluentHeaderView(QHeaderView):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setHighlightSections(False)
        self.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setStretchLastSection(True)
        self.setSectionsClickable(False)
        self.setStyleSheet(f"""
            QHeaderView {{
                background: transparent;
                border: none;
                margin: 0px;
                padding: 0px;
            }}
            QHeaderView::section {{
                background: {BG_COLOR};
                color: {TEXT_COLOR};
                border: none;
                padding: 6px 8px;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
            }}
            /* Round only the outermost sections to avoid seams */
            QHeaderView::section:horizontal:first {{
                border-top-left-radius: {RADIUS}px;
            }}
            QHeaderView::section:horizontal:last {{
                border-top-right-radius: {RADIUS}px;
            }}
        """)

class FluentTableWidget(QTableWidget):
    def __init__(self, rows=0, cols=2, parent=None):
        super().__init__(rows, cols, parent)

        # Basic view config
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)  # avoid double borders
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)

        header = FluentHeaderView(Qt.Horizontal, self)
        self.setHorizontalHeader(header)

        # Optional: if you have a vertical header sometimes visible, keep its bg consistent
        self.verticalHeader().setStyleSheet("QHeaderView::section { background: %s; border: none; }" % BG_COLOR)

        self.setStyleSheet(f"""
            QTableView {{
                background: white;
                border: 1px solid {PLACEHOLDER};
                border-radius: {RADIUS}px;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
                color: {TEXT_COLOR};
                alternate-background-color: #FAFAFA;
                outline: 0;
                selection-background-color: {ACCENT_COLOR};
                selection-color: white;
            }}
            QTableView::item {{
                padding: 4px 8px;
            }}
            QTableCornerButton::section {{
                background: {BG_COLOR};
                border: none;
                margin: 0px;
                padding: 0px;
            }}
            /* Fill the small square between header and vertical scrollbar */
            QAbstractScrollArea::corner {{
                background: {BG_COLOR};
                border: none;
            }}
        """)

class FluentRadioButton(QRadioButton):
    """
    Fluent-styled radio button with animated check/uncheck.
    - Keeps QRadioButton behavior (autoExclusive in QButtonGroup, keyboard nav).
    - Custom paint for indicator + text, hover/disabled states.
    - Small animation for inner dot grow/shrink.
    - Focus halo (the biggest outer ring) is REMOVED per request.
    """
    # geometry constants (you can tweak)
    INDICATOR_DIAM = 18   # outer circle diameter
    RING_THICKNESS = 2    # border thickness
    GAP_TEXT = 8          # space between indicator and text
    MARGIN_Y = 4          # top/bottom margins

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self.setMouseTracking(True)

        # Accent color can be overridden with set_color(...)
        self._accent = QColor(ACCENT_COLOR)
        self._hover = False

        # Progress used to animate inner-dot radius (0 -> 1)
        self._progress = 1.0 if self.isChecked() else 0.0
        self._anim = QPropertyAnimation(self, b"progress", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        # Keep appearance in sync when state changes
        self.toggled.connect(self._start_anim)

        # Focus policy (we keep keyboard focus for accessibility,
        # but we do NOT draw the big focus halo)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------- public API ----------
    def set_color(self, color):
        """Accepts '#RRGGBB' or QColor to override accent."""
        self._accent = QColor(color) if not isinstance(color, QColor) else color
        self.update()

    # ---------- animation ----------
    def _start_anim(self, on):
        self._anim.stop()
        self._anim.setStartValue(self._progress)
        self._anim.setEndValue(1.0 if on else 0.0)
        # Skip animation if disabled to match your ToggleSwitch behavior
        if not self.isEnabled():
            self._progress = 1.0 if on else 0.0
            self.update()
            return
        self._anim.start()

    def getProgress(self):
        return self._progress

    def setProgress(self, v):
        self._progress = float(v)
        self.update()

    progress = pyqtProperty(float, fget=getProgress, fset=setProgress)

    # ---------- events ----------
    def enterEvent(self, e):
        self._hover = True
        super().enterEvent(e)
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        super().leaveEvent(e)
        self.update()

    # ---------- sizing ----------
    def sizeHint(self):
        fm = self.fontMetrics()
        h_text = fm.height()
        h = max(self.INDICATOR_DIAM + self.MARGIN_Y * 2, h_text + self.MARGIN_Y * 2)
        w = self.INDICATOR_DIAM + self.GAP_TEXT + fm.horizontalAdvance(self.text()) + 4
        return QSize(w, h)

    # ---------- painting ----------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        r = self.rect()

        # Layout: indicator at left-center, text to the right
        ind_d = self.INDICATOR_DIAM
        ind_y = r.center().y() - ind_d // 2
        ind_x = r.x()
        ind_rect = QRect(ind_x, ind_y, ind_d, ind_d)

        # Colors based on state
        if not self.isEnabled():
            border_col = QColor(PLACEHOLDER)
            inner_col = QColor(BG_COLOR) if self.isChecked() else QColor(0, 0, 0, 0)
            text_col = QColor(PLACEHOLDER)
        else:
            text_col = QColor(TEXT_COLOR)
            if self.isChecked():
                border_col = self._accent
            else:
                # subtle hint on hover
                border_col = QColor(self._accent) if self._hover else QColor(PLACEHOLDER)
            inner_col = QColor(self._accent)
            # inner alpha depends on animation progress (for a soft feel)
            inner_col.setAlphaF(0.95 * self._progress + 0.05 if self.isChecked()
                                else 0.7 * self._progress)

        # ---- draw outer border ring (keep this; biggest focus halo is removed) ----
        pen = QPen(QColor(border_col))
        pen.setWidthF(self.RING_THICKNESS)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        outer_rect = QRectF(ind_rect).adjusted(
            self.RING_THICKNESS/2.0,
            self.RING_THICKNESS/2.0,
            -self.RING_THICKNESS/2.0,
            -self.RING_THICKNESS/2.0
        )
        painter.drawEllipse(outer_rect)

        # ---- draw inner dot (animated) ----
        if self._progress > 0.0:
            max_inner = ind_d - 6  # small padding to avoid touching ring
            d = max_inner * max(0.0, min(1.0, self._progress))  # float diameter
            inner_rect = QRectF(0.0, 0.0, d, d)
            inner_rect.moveCenter(outer_rect.center())
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(inner_col))
            painter.drawEllipse(inner_rect)

        # ---- NO big focus halo (removed) ----

        # ---- draw text ----
        painter.setPen(text_col)
        text_x = ind_rect.right() + self.GAP_TEXT
        text_rect = QRect(text_x, r.y(), r.width() - (text_x - r.x()), r.height())
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())

        painter.end()

class FluentWindow(FramelessWindow):
    hidden = pyqtSignal()

    def __init__(self,
                 use_fluent: bool = True,
                 ui_path: str = None,
                 widget_class: type = None,
                 widget_kwargs: dict = None,
                 title: str = ""):
        super().__init__()
        widget_kwargs = {**widget_kwargs, 'parent':self} or {'parent':self}

        # titlebar
        st = StandardTitleBar(self)
        st.setTitle(title)
        st.iconLabel.setFixedSize(0, 0)
        st.titleLabel.setStyleSheet(f"""
            QLabel{{
                background: transparent;
                font: {FONT_SIZE}pt "{FONT_FAMILY}";
                padding: 0 4px
            }}
        """)
        self.setTitleBar(st)

        if widget_class:
            self.loaded = widget_class(**widget_kwargs)
        elif ui_path:
            self.loaded = uic.loadUi(ui_path)
        else:
            raise ValueError("Must assign widget_class or ui_path")

        self.loaded.installEventFilter(self)
        # make sure bind this window to the close of loaded window
        # in case loaded create another window

        # FramelessWindow —
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 32, 0, 0)   # 32px for titlebar
        layout.addWidget(self.loaded)
        self.titleBar.raise_()

        self.loaded.adjustSize()
        self.resize(self.loaded.width(), self.loaded.height() + 32)
        self._hiding_via_close = False




    def eventFilter(self, obj, event):
        from .gui_device import LoadGUI
        if obj is self.loaded and isinstance(self.loaded, LoadGUI) and event.type() == QEvent.Hide:
            self.real_close()
            return False
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        event.ignore()
        self._hiding_via_close = True
        self.hide()

    def real_close(self):
        self.loaded.close()
        super().close()

    def hideEvent(self, event):
        if self._hiding_via_close:
            self.hidden.emit()
        super().hideEvent(event)

    def showEvent(self, event):
        self._hiding_via_close = False
        super().showEvent(event)


class FluentWindow_Q(QWidget):
    hidden = pyqtSignal()

    def __init__(self,
                 use_fluent: bool = True,
                 ui_path: str = None,
                 widget_class: type = None,
                 widget_kwargs: dict = None,
                 title: str = ""):
        super().__init__()
        widget_kwargs = widget_kwargs or {}

        self.setWindowTitle(title)

        if widget_class:
            self.loaded = widget_class(**widget_kwargs)
        elif ui_path:
            self.loaded = uic.loadUi(ui_path)
        else:
            raise ValueError("Must assign widget_class or ui_path")

        self.loaded.installEventFilter(self)
        # make sure bind this window to the close of loaded window
        # in case loaded create another window

        # QWindow
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.loaded)

        self.loaded.adjustSize()
        self.resize(self.loaded.width(), self.loaded.height())
        self._hiding_via_close = False

    def eventFilter(self, obj, event):
        from .gui_device import LoadGUI
        if obj is self.loaded and isinstance(self.loaded, LoadGUI) and event.type() == QEvent.Hide:
            self.real_close()
            return False
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        event.ignore()
        self._hiding_via_close = True
        self.hide()

    def real_close(self):
        self.loaded.close()
        super().close()

    def hideEvent(self, event):
        if self._hiding_via_close:
            self.hidden.emit()
        super().hideEvent(event)

    def showEvent(self, event):
        self._hiding_via_close = False
        super().showEvent(event)



global app_in_gui
# make sure the QApplication is keeped all the time
def run_fluent_window(use_fluent=True, widget_class=None, widget_kwargs=None, title='', ui_path=None, in_GUI=False, 
    window_handle=None):

    widget_kwargs = widget_kwargs or {}

    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        
    global app_in_gui
    app = QtWidgets.QApplication.instance()
    if app is None:
        app_in_gui = QtWidgets.QApplication(sys.argv)

    if window_handle is not None:
        fluent_widget = window_handle
    else:
        fluent_widget = FluentWindow(
        use_fluent=use_fluent,
        widget_class=widget_class,         
        widget_kwargs=widget_kwargs,     # params for __init__ widget
        title=title,
        ui_path = ui_path,
        )

    screen = app.primaryScreen()
    screen_geo = screen.availableGeometry()
    frame = fluent_widget.frameGeometry()
    frame.moveCenter(screen_geo.center())
    fluent_widget.move(frame.topLeft())
    # move to the center
    fluent_widget.show()

    if in_GUI is True:
        return fluent_widget
    # if called in gui then no exec_

    loop = QtCore.QEventLoop()
    fluent_widget.hidden.connect(loop.quit)
    loop.exec_()
    return fluent_widget


class DynamicPlainTextEdit(FluentPlainTextEdit):
    """QPlainTextEdit that auto-adjusts its height to fit content exactly, hides scrollbars."""
    def __init__(self, text='', placeholder_text = '', parent=None):
        super().__init__(parent)
        self.setPlainText(text)
        self.setPlaceholderText(placeholder_text)

        # Keep scrollbars hidden as in your original code.
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # NEW: enable soft wrap so long lines wrap instead of overflowing horizontally.
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)

        self._doc_margin = self.document().documentMargin()

        # Recompute height when text changes and when layout size changes due to wrapping.
        self.textChanged.connect(self._adjust_height)
        self.document().documentLayout().documentSizeChanged.connect(lambda *_: self._adjust_height())

        self._adjust_height()

    def resizeEvent(self, e):
        """Recompute height when width changes (wrapping depends on the viewport width)."""
        super().resizeEvent(e)
        self._adjust_height()

    def _adjust_height(self):
        """Keep your original height formula but count *visual* lines (includes soft wraps)."""
        # --- compute visual line count ---
        doc = self.document()
        # Make wrapping depend on the current viewport width.
        doc.setTextWidth(self.viewport().width())
        # Force relayout so line counts are up-to-date before we read them.
        doc.adjustSize()

        line_count = 0
        dl = doc.documentLayout()
        block = doc.begin()
        while block.isValid():
            # Touch the block to ensure its layout is created.
            if dl is not None:
                dl.blockBoundingRect(block)
            bl = block.layout()
            line_count += (bl.lineCount() if bl else 1)
            block = block.next()
        line_count = max(1, line_count)  # at least one line

        # --- your original math below ---
        fm = self.fontMetrics()
        line_height = fm.lineSpacing()
        frame = self.frameWidth() * 2
        margin = self._doc_margin * 2

        total = int(line_count * line_height + frame + margin)
        self.setFixedHeight(total)
        self.updateGeometry()