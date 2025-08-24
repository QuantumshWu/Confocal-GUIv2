import os, sys, time
import threading
import re
import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings
from scipy.optimize import OptimizeWarning


def dummy_cross(ax, x, y):

    def data_to_display(ax, xdata, ydata):
        return ax.transData.transform((xdata, ydata))

    x_disp, y_disp = data_to_display(ax, x, y)

    press_event = MouseEvent(name='button_press_event',
                         canvas=ax.figure.canvas,
                         x=x_disp, y=y_disp,
                         button=3)  
    press_event.inaxes = ax  
    ax.figure.canvas.callbacks.process('button_press_event', press_event)
    time.sleep(0.31)

    ax.figure.canvas.callbacks.process('button_press_event', press_event)



def dummy_area(ax, x1, y1, x2, y2):

    x1_disp, y1_disp = ax.transData.transform((x1, y1))
    x2_disp, y2_disp = ax.transData.transform((x2, y2))

    press_event = MouseEvent('button_press_event', ax.figure.canvas, x1_disp, y1_disp, button=1)
    press_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_press_event', press_event)
    motion_event = MouseEvent('motion_notify_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    motion_event.inaxes = ax
    ax.figure.canvas.callbacks.process('motion_notify_event', motion_event)
    release_event = MouseEvent('button_release_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    release_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_release_event', release_event)
    # the first selection close existing rectangle, otherwise bug
    time.sleep(0.01)
    press_event = MouseEvent('button_press_event', ax.figure.canvas, x1_disp, y1_disp, button=1)
    press_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_press_event', press_event)
    motion_event = MouseEvent('motion_notify_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    motion_event.inaxes = ax
    ax.figure.canvas.callbacks.process('motion_notify_event', motion_event)
    release_event = MouseEvent('button_release_event', ax.figure.canvas, x2_disp, y2_disp, button=1)
    release_event.inaxes = ax
    ax.figure.canvas.callbacks.process('button_release_event', release_event)

            


valid_fit_func = ['lorent', 'decay', 'rabi', 'lorent_zeeman', 'center']         
class DataFigure():
    """
    The class contains all data of the figure, enables more operations
    such as curve fit or save data
    
    Parameters
    ----------
    live_plot :instance of class LivePlot
    
    
    Examples
    --------
    >>> data_figure = DataFigure(live_plot=live_plot)
    or
    >>> data_figure = DataFigure(is_GUI=True)
    
    >>> data_x, data_y = data_figure.data
    
    >>> data_figure.save('my_figure')
    'save to my_figure_{time}.jpg and my_figure_{time}.txt'

    >>> data_figure.lorent(p0 = None)
    'figure with lorent curve fit'
    
    >>> data_figure.clear()
    'remove lorent fit and text'
    """
    def __init__(self, live_plot=None):

        self.live_plot = live_plot
        self.data_x = live_plot.data_x
        self.data_x_original = copy.deepcopy(live_plot.data_x)
        self.data_y = live_plot.data_y
        self.fig = self.live_plot.fig
        self.area = self.live_plot.area
        self.zoom = self.live_plot.zoom
        self.cross = self.live_plot.cross
        self.ylabel_original = self.live_plot.ylabel

        first_ax = self.fig.axes[0]
        if any(isinstance(im, matplotlib.image.AxesImage) for im in first_ax.get_images()):
            self.plot_type = '2D'
        else:
            self.plot_type = '1D'

        self.p0 = None
        self.fit = None
        self.fit_func = None
        self.text = None
        self.info = self.live_plot.controller.measurement.info
        self.measurement_name = self.live_plot.controller.measurement.name
        self._load_unit()
        warnings.filterwarnings("ignore", category=OptimizeWarning)

    def _load_unit(self):

        x_label = self.live_plot.xlabel
        pattern = r'\((.+)\)$'
        match = re.search(pattern, x_label)
        if match:
            self.unit = match.group(1)
        else:
            self.unit = '1'

        self.unit_original = self.live_plot.controller.measurement.unit

        if self.unit in ['GHz', 'nm', 'MHz']:
            spl = 299792458  # m/s
            self.conversion_map = {
                'nm': ('GHz', lambda x: spl / x),
                'GHz': ('MHz', lambda x: x * 1e3),
                'MHz': ('nm', lambda x: spl / (x/1e3))
            }
        elif self.unit in ['ns', 'us', 'ms']:
            self.conversion_map = {
                'ms': ('ns', lambda x: x * 1e6),
                'ns': ('us', lambda x: x / 1e3),
                'us': ('ms', lambda x: x / 1e3)
            }
        else:
            self.conversion_map = None

        self._update_transform_back()
        

    def xlim(self, x_min, x_max):
        self.fig.axes[0].set_xlim(x_min, x_max)

    def ylim(self, y_min, y_max):
        self.fig.axes[0].set_ylim(y_min, y_max)


    def _align_to_grid(self, x, type):
        # round to center of 2D grid
        # type='x' for x, type='y' for y
        if not self.plot_type == '2D':
            return
        if not hasattr(self, 'grid_center'):
            self.grid_center = self.data_x[0] # one of the center of grid, [x_center, y_center]
            self.step_x = np.abs(self.live_plot.x_array[1] - self.live_plot.x_array[0])
            self.step_y = np.abs(self.live_plot.y_array[1] - self.live_plot.y_array[0])

        if type == 'x':
            return round((x-self.grid_center[0])/self.step_x)*self.step_x + self.grid_center[0]
        if type == 'y':
            return round((x-self.grid_center[1])/self.step_y)*self.step_y + self.grid_center[1]

    def save(self, addr='', extra_info=None):
        current_time = time.localtime()
        current_date = time.strftime("%Y-%m-%d", current_time)
        current_time_formatted = time.strftime("%H:%M:%S", current_time)
        time_str = current_date.replace('-', '_') + '_' + current_time_formatted.replace(':', '_')

        if addr=='' or ('/' not in addr):
            pass
        else:
            directory = os.path.dirname(addr)
            if not os.path.exists(directory):
                os.makedirs(directory)

        if extra_info is None:
            extra_info = {}
        info = {**self.info, **extra_info, 'labels':self.live_plot.labels, 
                'points_done':self.live_plot.controller.measurement.points_done,
                'repeat_cur':self.live_plot.controller.measurement.repeat_cur}

        filename = '_'.join(p for p in (self.measurement_name, time_str) if p)
        is_dirlike = bool(addr) and (addr.endswith(('/', '\\')) or os.path.isdir(addr))
        base = os.path.join(addr, filename) if is_dirlike \
               else '_'.join(p for p in (addr, filename) if p)
        # addr="/home/u/run1/" → /home/u/run1/name_time
        # addr="xxx/" → xxx/name_time
        # addr="xxx" → xxx_name_time
        # addr="" → name_time
        # addr="xxx/yyy" → xxx/yyy_name_time

        self.fig.savefig(f'{base}.jpg')
        np.savez(f'{base}.npz', data_x = self.data_x_original, data_y = self.data_y, \
            info = info)
        # here save the data state before any change_unit in data_figure
        print(f'saved fig as {base}.npz')
        return f'{base}.npz'

    def _min_overlap(self, ax, text, candidates=None):
        """
        Given a list of candidate positions in normalized coordinates, each candidate being a tuple:
            (norm_x, norm_y, ha, va),
        this function positions the text (using ax.transAxes) for each candidate, forces a draw,
        and calculates the fraction of the line's total length that overlaps with the text's bounding box.
        For multi-line text (detected via '\n'), the function computes an overall bounding box
        from the rendered sizes of the first line and the remaining lines.

        The Liang-Barsky algorithm is used to compute the intersection length between each line segment
        (formed by consecutive data points) and the candidate text bounding box.

        The candidate whose overlapping fraction (overlap length / total line length) is minimal is chosen.

        Parameters:
            ax : matplotlib.axes.Axes
                The axes that contain the line and text.
            text : matplotlib.text.Text
                The text object to be positioned.
            candidates : list of tuples
                A list of candidate positions, each specified as 
                (normalized_x, normalized_y, ha, va), where normalized_x and normalized_y are in [0, 1]
                (axes coordinates), and ha, va are the horizontal and vertical alignment strings.
        """

        # Use default candidates if none provided.
        if candidates is None:
            candidates = [
                (0.025, 0.85, 'left', 'top'),
                (0.975, 0.85, 'right', 'top'),
                (0.025, 0.025, 'left', 'bottom'),
                (0.975, 0.025, 'right', 'bottom'),
                (0.025, 0.5, 'left', 'center'),
                (0.975, 0.5, 'right', 'center'),
                (0.5, 0.025, 'center', 'bottom'),
                (0.5, 0.85, 'center', 'top'),
                (0.5, 0.5, 'center', 'center'),
            ]

        canvas = ax.figure.canvas
        # Hide text during processing.
        text.set_alpha(0)
        orig_text = text.get_text()
        renderer = canvas.get_renderer()

        # ---------------------------------------------------------------------
        # Precompute polyline points in display coordinates.
        if self.plot_type == '1D':
            pts = np.column_stack([self.data_x_p, self.data_y_p])
            pts_disp = ax.transData.transform(pts)
            pts_full = np.column_stack([self.data_x[:, 0], self.data_y[:, 0]])
            pts_disp_full = ax.transData.transform(pts_full)
        elif self.plot_type == '2D':
            pts = [[self.popt[-2], self.popt[-1]], [self.popt[-2]+1e-3, self.popt[-1]+1e-3]]
            pts_disp = ax.transData.transform(pts)
            pts_full = [[self.popt[-2], self.popt[-1]], [self.popt[-2]+1e-3, self.popt[-1]+1e-3]]
            pts_disp_full = ax.transData.transform(pts_full)
            # set center of fit as the line which text needed to avoid


        def total_length(pts_arr):
            """Compute the total length of a polyline given its display coordinates."""
            seg_lengths = np.hypot(np.diff(pts_arr[:, 0]), np.diff(pts_arr[:, 1]))
            return np.sum(seg_lengths)

        total_length_par = total_length(pts_disp)
        total_length_full = total_length(pts_disp_full)

        # ---------------------------------------------------------------------
        # Precompute overall text bounding box dimensions.
        # Render the text at the center (with center alignment) to obtain a consistent size.
        text.set_ha('center')
        text.set_va('center')
        text.set_position((0.5, 0.5))
        lines = orig_text.split("\n")

        # Render the first line and get its bounding box.
        text.set_text(lines[0])
        canvas.draw()
        first_bbox = text.get_window_extent(renderer)
        first_width = first_bbox.width
        first_height = first_bbox.height

        # If multi-line text, render the remaining lines and get their bounding box.
        if len(lines) > 1:
            text.set_text("\n".join(lines[1:]))
            canvas.draw()
            rest_bbox = text.get_window_extent(renderer)
            rest_width = rest_bbox.width
            rest_height = rest_bbox.height
        else:
            rest_bbox = None
            rest_width = 0
            rest_height = 0

        overall_width = first_width if rest_bbox is None else max(first_width, rest_width)
        overall_height = first_height if rest_bbox is None else (first_height + rest_height)

        # ---------------------------------------------------------------------
        # Helper: Compute candidate bounding box in display coordinates.
        def candidate_bbox(norm_x, norm_y, ha, va):
            """
            Compute the candidate text bounding box (xmin, ymin, xmax, ymax) in display coordinates,
            given normalized coordinates and alignment.
            """
            anchor_disp = ax.transAxes.transform((norm_x, norm_y))
            # Horizontal alignment.
            if ha == 'left':
                bbox_x0 = anchor_disp[0]
            elif ha == 'center':
                bbox_x0 = anchor_disp[0] - overall_width / 2
            elif ha == 'right':
                bbox_x0 = anchor_disp[0] - overall_width
            else:
                bbox_x0 = anchor_disp[0]

            # Vertical alignment.
            if va == 'top':
                bbox_y1 = anchor_disp[1]
                bbox_y0 = bbox_y1 - overall_height
            elif va == 'center':
                bbox_y0 = anchor_disp[1] - overall_height / 2
                bbox_y1 = anchor_disp[1] + overall_height / 2
            elif va == 'bottom':
                bbox_y0 = anchor_disp[1]
                bbox_y1 = bbox_y0 + overall_height
            else:
                bbox_y0 = anchor_disp[1] - overall_height / 2
                bbox_y1 = anchor_disp[1] + overall_height / 2

            return (bbox_x0, bbox_y0, bbox_x0 + overall_width, bbox_y0 + overall_height)

        # ---------------------------------------------------------------------
        # Liang-Barsky algorithm to compute overlapping length.
        def liang_barsky_clip_length(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
            """
            Compute the length of the portion of the line segment from (x0,y0) to (x1,y1)
            that lies within the rectangle [xmin, xmax] x [ymin, ymax] using the Liang-Barsky algorithm.
            """
            dx = x1 - x0
            dy = y1 - y0
            p = [-dx, dx, -dy, dy]
            q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]
            u1, u2 = 0.0, 1.0

            for pi, qi in zip(p, q):
                if pi == 0:
                    if qi < 0:
                        return 0.0  # Line is parallel and outside the boundary.
                else:
                    t = qi / float(pi)
                    if pi < 0:
                        u1 = max(u1, t)
                    else:
                        u2 = min(u2, t)
            if u1 > u2:
                return 0.0  # No valid intersection.
            seg_length = np.hypot(dx, dy)
            return (u2 - u1) * seg_length

        def compute_overlap_length(pts_arr, rect):
            """
            Compute the total overlapping length of the polyline (represented by pts_arr)
            with the given rectangular region.
            """
            xmin, ymin, xmax, ymax = rect
            overlap = 0.0
            for i in range(len(pts_arr) - 1):
                x0, y0 = pts_arr[i]
                x1, y1 = pts_arr[i+1]
                overlap += liang_barsky_clip_length(x0, y0, x1, y1, xmin, xmax, ymin, ymax)
            return overlap

        # ---------------------------------------------------------------------
        # Iterate over candidate positions and compute overlapping fractions.
        best_candidate_par = None
        min_fraction_par = np.inf
        best_candidate_full = None
        min_fraction_full = np.inf
        overlap_fraction_par_for_full_candidate = None

        for candidate in candidates:
            norm_x, norm_y, ha, va = candidate
            rect = candidate_bbox(norm_x, norm_y, ha, va)

            # (Optional) Update text properties for visualization.
            text.set_ha(ha)
            text.set_va(va)
            text.set_position((norm_x, norm_y))
            text.set_text(orig_text)
            canvas.draw()  # This draw() may be removed if not strictly necessary.

            overlap_par = compute_overlap_length(pts_disp, rect)
            overlap_full = compute_overlap_length(pts_disp_full, rect)

            fraction_par = overlap_par / total_length_par if total_length_par > 0 else 0
            fraction_full = overlap_full / total_length_full if total_length_full > 0 else 0

            if fraction_par < min_fraction_par:
                min_fraction_par = fraction_par
                best_candidate_par = candidate

            if fraction_full < min_fraction_full:
                min_fraction_full = fraction_full
                best_candidate_full = candidate
                overlap_fraction_par_for_full_candidate = fraction_par

        # Choose best candidate: if candidate from "full" set yields zero overlap in "par", prefer it.
        best_candidate = best_candidate_full if overlap_fraction_par_for_full_candidate == 0 else best_candidate_par

        # ---------------------------------------------------------------------
        # Update text object with the best candidate and make it visible.
        norm_x, norm_y, ha, va = best_candidate
        text.set_ha(ha)
        text.set_va(va)
        text.set_position((norm_x, norm_y))
        text.set_text(orig_text)
        text.set_alpha(1)
        canvas.draw()



    def _display_popt(self, popt, popt_str):
        # popt_str = ['amplitude', 'offset', 'omega', 'decay', 'phi'], popt_pos = 'lower left' etc

        _popt = popt
        formatted_popt = [f'{x:.5f}'.rstrip('0') for x in _popt]
        result_list = [f'{name}={value}' for name, value in zip(popt_str, formatted_popt)]
        formatted_popt_str = '\n'.join(result_list)
        result = f"{self.formula_str}\n{formatted_popt_str}"
                    
        if self.text is None:
            self.text = self.fig.axes[0].text(0.5, 0.5, 
                                              result, transform=self.fig.axes[0].transAxes, 
                                              color='blue', ha='center', va='center', fontsize=matplotlib.rcParams['legend.fontsize'])

        else:
            self.text.set_text(result)


        self._min_overlap(self.fig.axes[0], self.text)
        for line in self.live_plot.lines:
            line.set_alpha(0.5)

        self.fig.canvas.draw()

    def _select_fit(self, min_num=2):
        # return data in the area selector, and only return first set if there are multiple sets of data (only data not data_ref)
        valid_index = [i for i, data in enumerate(self.data_y) if not np.isnan(data[0])]
        # index of none np.nan data
        if self.plot_type == '1D':
            if self.area.range[0] is None:
                xlim = self.fig.axes[0].get_xlim()
                index_l = np.argmin(np.abs(self.data_x[valid_index, 0] - xlim[0]))
                index_h = np.argmin(np.abs(self.data_x[valid_index, 0] - xlim[1]))
                index_l, index_h = np.sort([index_l, index_h])
                # in order to handle data_x from max to min (e.g. GHz unit)
                if np.abs(index_l - index_h)<=min_num:
                    return self.data_x[valid_index, 0], self.data_y[valid_index, 0]
                return self.data_x[valid_index, 0][index_l:index_h], self.data_y[valid_index, 0][index_l:index_h]
            else:
                xl, xh, yl, yh = self.area.range
                if (xl is None) or (xh is None):
                    return self.data_x[valid_index, 0], self.data_y[valid_index, 0]
                if (xl - xh)==0:
                    return self.data_x[valid_index, 0], self.data_y[valid_index, 0]

                index_l = np.argmin(np.abs(self.data_x[valid_index, 0] - xl))
                index_h = np.argmin(np.abs(self.data_x[valid_index, 0] - xh))
                index_l, index_h = np.sort([index_l, index_h])
                # in order to handle data_x from max to min (e.g. GHz unit)
                if np.abs(index_l - index_h)<=min_num:
                    return self.data_x[valid_index, 0], self.data_y[valid_index, 0]
                return self.data_x[valid_index, 0][index_l:index_h], self.data_y[valid_index, 0][index_l:index_h]

        elif self.plot_type == '2D':
            if self.area.range[0] is None:
                xl, xh = np.sort(self.fig.axes[0].get_xlim())
                yl, yh = np.sort(self.fig.axes[0].get_ylim())
                xl, xh = [self._align_to_grid(v, 'x') for v in (xl, xh)]
                yl, yh = [self._align_to_grid(v, 'y') for v in (yl, yh)]
                index_area = np.where(
                    (self.data_x[valid_index, 0] >= xl) & (self.data_x[valid_index, 0] <= xh) &
                    (self.data_x[valid_index, 1] >= yl) & (self.data_x[valid_index, 1] <= yh)
                )[0]
                if len(index_area)<=min_num:
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]

                data_x_p = self.data_x[valid_index][index_area]
                return (data_x_p[:, 0], data_x_p[:, 1]), self.data_y[valid_index][index_area, 0]
            else:
                xl, xh, yl, yh = self.area.range
                if (xl is None) or (xh is None):
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]
                if (xl - xh)==0:
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]

                xl, xh = [self._align_to_grid(v, 'x') for v in (xl, xh)]
                yl, yh = [self._align_to_grid(v, 'y') for v in (yl, yh)]
                index_area = np.where(
                    (self.data_x[valid_index, 0] >= xl) & (self.data_x[valid_index, 0] <= xh) &
                    (self.data_x[valid_index, 1] >= yl) & (self.data_x[valid_index, 1] <= yh)
                )[0]
                if len(index_area)<=min_num:
                    return (self.data_x[valid_index][:, 0], self.data_x[valid_index][:, 1]), self.data_y[valid_index, 0]

                data_x_p = self.data_x[valid_index][index_area]
                return (data_x_p[:, 0], data_x_p[:, 1]), self.data_y[valid_index][index_area, 0]
                # should return data_x_p as ([x0, x1, ...], [y0, y1, ...])

       

    def _fit_and_draw(self, is_fit, is_display, kwargs):
        # use self.p0_list and self.bounds, self.popt_str, self._fit_func
        for index, param in enumerate(self.popt_str):
            clean = re.sub(r'[\\$]', '', param)
            param_in = kwargs.get(clean, None)
            if param_in is not None:
                self.bounds[0][index], self.bounds[1][index] = np.sort([param_in*(1-1e-5), param_in*(1+1e-5)])
                for p0 in self.p0_list:
                    p0[index] = param_in

        if is_fit:
            try:
                loss_min = np.inf
                for p0 in self.p0_list:
                    popt_cur, pcov_cur = curve_fit(self._fit_func, self.data_x_p, self.data_y_p, p0=p0, bounds = self.bounds)
                    loss_cur = np.sum((self._fit_func(self.data_x_p, *popt_cur) - self.data_y_p)**2)
                    if loss_cur<loss_min:
                        loss_min = loss_cur
                        popt = popt_cur
                        pcov = pcov_cur
            except:
                return 'error', 'error'

        else:
            popt, pcov = self.p0_list[0], None
        self.popt = popt

        if is_display:
            self._display_popt(popt, self.popt_str)

        if self.plot_type == '1D':
            if self.fit is None:
                self.fit = self.fig.axes[0].plot(self.data_x[:, 0], self._fit_func(self.data_x[:, 0], *popt), color='orange', linestyle='--')
            else:
                self.fit[0].set_ydata(self._fit_func(self.data_x[:, 0], *popt))
        elif self.plot_type == '2D':
            if self.fit is None:
                self.fit = [self.fig.axes[0].scatter(popt[-2], popt[-1], color='orange', s=50),]
                circle = matplotlib.patches.Circle((popt[-2], popt[-1]), radius=popt[-3], edgecolor='orange'
                    , facecolor='none', linewidth=2, alpha=0.5)
                self.fit.append(circle)
                self.fig.axes[0].add_patch(circle)
            else:
                self.fit[0].set_offsets((popt[-2], popt[-1]))
                self.fit[1].set_center((popt[-2], popt[-1]))
                self.fit[1].set_radius(popt[-3])

        self.fig.canvas.draw()

        return popt, pcov


    def lorent(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '2D':
            return [None, None], None
        self.data_x_p, self.data_y_p = self._select_fit(min_num=4)
        # use the area selector results for fitting , min_num should at least be number of fitting parameters
        self.formula_str = '$f(x)=H\\frac{(FWHM/2)^2}{(x-x_0)^2+(FWHM/2)^2}+B$'
        def _lorent(x, center, full_width, height, bg):
            return height*((full_width/2)**2)/((x - center)**2 + (full_width/2)**2) + bg
        self._fit_func = _lorent
        if p0 is None:# no input
            self.p0_list = []
            # likely be positive height
            guess_center = self.data_x_p[np.argmax(self.data_y_p)]
            guess_height = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_bg = np.min(self.data_y_p)
            guess_full_width = np.abs(self.data_x_p[0]-self.data_x_p[-1])/4
            self.p0_list.append([guess_center, guess_full_width, guess_height, guess_bg])

            # likely be negtive height
            guess_center = self.data_x_p[np.argmin(self.data_y_p)]
            guess_height = -np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_bg = np.max(self.data_y_p)
            guess_full_width = np.abs(self.data_x_p[0]-self.data_x_p[-1])/4
            self.p0_list.append([guess_center, guess_full_width, guess_height, guess_bg])


        else:
            self.p0_list = [p0, ]
            guess_center = self.p0[0]
            guess_full_width = self.p0[1]
            guess_height = self.p0[2]
            guess_bg = self.p0[3]

        data_x_range = np.abs(self.data_x_p[-1] - self.data_x_p[0])
        data_y_range = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
        self.bounds = ([np.nanmin(self.data_x_p), guess_full_width/10, -10*data_y_range, np.nanmin(self.data_y_p)-10*data_y_range], \
        [np.nanmax(self.data_x_p), guess_full_width*10, 10*data_y_range, np.nanmax(self.data_y_p)+10*data_y_range])
        
        self.popt_str = ['$x_0$', 'FWHM', 'H', 'B']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = 'lorent'
        return [self.popt_str, pcov], popt


    def lorent_zeeman(self, p0=None, is_display=True, is_fit=True, **kwargs):
        #fit of D1 under B field, will rewrite soon
        if self.plot_type == '2D':
            return [None, None], None 
        
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        # use the area selector results for fitting , min_num should at least be number of fitting parameters
        self.formula_str = '$f(x)=H(L(\\delta/2)+L(-\\delta/2))+B$'
        def _lorent_zeeman(x, center, full_width, height, bg, split):
            return height*((full_width/2)**2)/((x - center - split/2)**2 + (full_width/2)**2) \
                + height*((full_width/2)**2)/((x - center + split/2)**2 + (full_width/2)**2) + bg
        self._fit_func = _lorent_zeeman
        if p0 is None:# no input
            self.p0_list = []
            try:
                guess_height = (np.max(self.data_y_p)-np.min(self.data_y_p))
                peaks, properties = find_peaks(self.data_y_p, width=1, prominence=guess_height/8) # width about 100MHz
                if len(peaks)==0:
                    return
                peaks_largest = peaks[np.argsort(self.data_y_p[peaks])[::-1]]
                for second_peak in peaks_largest:
                    guess_center = self.data_x_p[int(np.mean([peaks_largest[0], second_peak]))]
                    guess_full_width = properties['widths'][np.argsort(self.data_y_p[peaks])[-1]]*np.abs(self.data_x_p[1]-self.data_x_p[0])
                    guess_spl = np.abs((self.data_x_p[second_peak] - guess_center)*2)
                    if guess_spl<guess_full_width:
                        guess_height = guess_height/2
                    guess_bg = np.min(self.data_y_p)
                    self.p0_list.append([guess_center, guess_full_width, guess_height, guess_bg, guess_spl])

            except:
                return
        else:
            self.p0_list = [p0, ]
            guess_center = self.p0[0]
            guess_full_width = self.p0[1]
            guess_height = self.p0[2]
            guess_bg = self.p0[3]

        data_x_range = np.abs(self.data_x_p[-1] - self.data_x_p[0])
        data_y_range = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
        self.bounds = ([np.nanmin(self.data_x_p), guess_full_width/10, -10*data_y_range, np.nanmin(self.data_y_p)-10*data_y_range, 0], \
        [np.nanmax(self.data_x_p), guess_full_width*10, 10*data_y_range, np.nanmax(self.data_y_p)+10*data_y_range, 2*data_x_range])
        
        self.popt_str = ['$x_0$', 'FWHM', 'H', 'B', '$\\delta$']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = 'lorent_zeeman'
        return [self.popt_str, pcov], popt


    def rabi(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '2D':
            return [None, None], None 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)

        self.formula_str = '$f(x)=A\\sin(2{\\pi}fx+\\varphi)e^{-x/\\tau}+B$'
        def _rabi(x, amplitude, offset, omega, decay, phi):
            return amplitude*np.sin(2*np.pi*omega*x + phi)*np.exp(-x/decay) + offset
        self._fit_func = _rabi
        if p0 is None:# no input
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))/2
            guess_offset = np.mean(self.data_y_p)


            N = len(self.data_y_p)
            delta_x = self.data_x_p[1] - self.data_x_p[0]
            y_detrended = self.data_y_p - np.mean(self.data_y_p)
            fft_vals = np.fft.fft(y_detrended)
            fft_freq = np.fft.fftfreq(N, d=delta_x)
            mask = fft_freq > 0
            fft_vals = fft_vals[mask]
            fft_freq = fft_freq[mask]
            idx_peak = np.argmax(np.abs(fft_vals))
            guess_omega = fft_freq[idx_peak]
            # fft data to get frequency 

            delta_x_min_max = np.abs(self.data_x_p[np.argmin(self.data_y_p)] - self.data_x_p[np.argmax(self.data_y_p)])
            ratio_min_max = (np.abs(np.min(self.data_y_p) - guess_offset)/np.abs(np.max(self.data_y_p) - guess_offset))
            # amp_min = amp_max*exp(-delta_x_min_max/guess_decay)
            # guess_decay = -delta_x_min_max/ln(ratio_min_max)
            guess_decay = np.abs(-delta_x_min_max/np.log(ratio_min_max))
            guess_phi = np.pi/2


            self.p0_list = [[guess_amplitude, guess_offset, guess_omega, guess_decay, guess_phi],]

        else:
            self.p0_list = [p0, ]

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_omega = self.p0[2]
            guess_decay = self.p0[3]
            guess_phi = self.p0[4]

        data_y_range = np.max(self.data_y_p) - np.min(self.data_y_p)
        self.bounds = ([guess_amplitude/5, np.nanmin(self.data_y_p), guess_omega/5, guess_decay/5, guess_phi - np.pi/20], \
            [guess_amplitude*5, np.nanmax(self.data_y_p), guess_omega*5, guess_decay*5, guess_phi + np.pi/20])
        
        self.popt_str = ['A', 'B', 'f', '$\\tau$', '$\\varphi$']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)            
        self.fit_func = 'rabi'
        return [self.popt_str, pcov], popt


    def decay(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '2D':
            return [None, None], None 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=3)
        self.formula_str = '$f(x)=Ae^{-x/\\tau}+B$'
        def _exp_decay(x, amplitude, offset, decay):
            return amplitude*np.exp(-x/decay) + offset
        self._fit_func = _exp_decay
        if p0 is None:# no input
            self.p0_list = []
            # if positive
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_offset = np.mean(self.data_y_p)
            guess_decay = np.abs(self.data_x_p[np.argmin(self.data_y_p)] - self.data_x_p[np.argmax(self.data_y_p)])/2
            self.p0_list.append([guess_amplitude, guess_offset, guess_decay])
            # if negtive
            self.p0_list.append([-guess_amplitude, guess_offset, guess_decay])

        else:
            self.p0_list = [p0, ]

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_decay = self.p0[2]
            
        data_y_range = np.max(self.data_y_p) - np.min(self.data_y_p)
        self.bounds = ([-4*data_y_range, guess_offset - data_y_range, guess_decay/10], \
            [4*data_y_range, guess_offset + data_y_range, guess_decay*10])
        
        self.popt_str = ['A', 'B','$\\tau$']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)            
        self.fit_func = 'decay'
        return [self.popt_str, pcov], popt

    # 2D plot fit only display center, and x0, y0 must be last two parameters for fit_func
    def center(self, p0=None, is_display=True, is_fit=True, **kwargs):
        if self.plot_type == '1D':
            return [None, None], None 
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        # data_x_p is [[x0, y0], [x1, y1], ...]
        self.formula_str = '$f(r)=Ae^{-(r-(x0,y0))^2/R^2}+B$'
        def _center(coord, amplitude, offset, size, x0, y0):
            # coord is (x_array, y_array) or (x, y)
            # center is (x0, y0)
            x, y = coord
            x, y = np.array(x), np.array(y)
            x_dis = np.abs(x - x0)
            y_dis = np.abs(y - y0)
            return amplitude*np.exp(-(x_dis**2+y_dis**2)/size**2) + offset

        self._fit_func = _center
        if p0 is None:# no input
            self.p0_list = []
            guess_amplitude = np.abs(np.max(self.data_y_p) - np.min(self.data_y_p))
            guess_offset = np.mean(self.data_y_p)

            max_5_points = np.argsort(self.data_y_p)[::-1][:5]
            x_range = np.ptp(self.data_x_p[0][max_5_points])
            y_range = np.ptp(self.data_x_p[1][max_5_points])
            guess_size = np.hypot(x_range, y_range)
            guess_x0 = np.mean(self.data_x_p[0][max_5_points])
            guess_y0 = np.mean(self.data_x_p[1][max_5_points])

            self.p0_list.append([guess_amplitude, guess_offset, guess_size, guess_x0, guess_y0])

        else:
            self.p0_list = [p0, ]

            guess_amplitude = self.p0[0]
            guess_offset = self.p0[1]
            guess_size = self.p0[2]
            guess_x0 = self.p0[3]
            guess_y0 = self.p0[4]

        self.bounds = ([guess_amplitude/5, 0, guess_size/10, np.min(self.data_x_p[0]), np.min(self.data_x_p[1])], \
            [guess_amplitude*5, 2*guess_offset, guess_size*10, np.max(self.data_x_p[0]), np.max(self.data_x_p[1])])
        self.popt_str = ['A', 'B', 'R', 'x0', 'y0']
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)            
        self.fit_func = 'center'
        return [self.popt_str, pcov], popt


            
    def clear(self):
        if (self.text is None) and (self.fit is None):
            return
        if self.text is not None:
            self.text.remove()
        if self.fit is not None:
            for fit in self.fit:
                fit.remove()
        for line in self.live_plot.lines:
            line.set_alpha(1)
        self.fig.canvas.draw()
        self.fit = None
        self.text = None

    def _update_unit(self, transform):

        for line in self.fig.axes[0].lines:
            data_x = np.array(line.get_xdata())
            if np.array_equal(data_x, np.array([0, 1])):
                line.set_xdata(data_x)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    new_xdata = np.where(data_x != 0, transform(data_x), np.inf)
                    line.set_xdata(new_xdata)

        xlim = self.fig.axes[0].get_xlim()
        self.data_x = self.fig.axes[0].lines[0].get_xdata().reshape(-1, 1)
        self.fig.axes[0].set_xlim(transform(xlim[0]), transform(xlim[-1]))

        if self.area.range[0] is not None:
            new_x1 = transform(self.area.range[0])
            new_x2 = transform(self.area.range[1])
            new_y1 = self.area.range[2]
            new_y2 = self.area.range[3]
            dummy_area(self.area.ax, new_x1, new_y1, new_x2, new_y2)

        if self.cross.xy is not None:
            new_x = transform(self.cross.xy[0])
            new_y = self.cross.xy[1]
            dummy_cross(self.cross.ax, new_x, new_y) 

        if self.fit is not None:
            self.clear()
            try:
                exec(f'self.{self.fit_func}()')
            except:
                pass

    def change_unit(self):
        if (self.plot_type == '2D') or (self.conversion_map is None):
            return

        new_unit, conversion_func = self.conversion_map[self.unit]

        ax = self.fig.axes[0]
        old_xlabel = ax.get_xlabel()
        new_xlabel = re.sub(r'\((.+)\)$', f'({new_unit})', old_xlabel)
        self.fig.axes[0].set_xlabel(new_xlabel)
        self.unit = new_unit
        self._update_transform_back()
        self._update_unit(conversion_func)
        # update selector after transform_back to avoid wrong unit read with callback
        self.fig.canvas.draw()

    def _update_transform_back(self):
        import functools
        transforms = []
        temp_unit = self.unit
        while (self.conversion_map is not None) and (temp_unit != self.unit_original):
            try:
                next_unit, conv_func = self.conversion_map[temp_unit]
            except KeyError:
                print(f'Unit {temp_unit} not in conversion_map')
                break
            transforms.append(conv_func)
            temp_unit = next_unit

        self.transform_back = (lambda x: functools.reduce(lambda a, f: f(a), transforms, x)) if transforms else lambda x: x





 



















