import pyqtgraph as pg
from PyQt6 import QtCore

class EnhancedViewBox(pg.ViewBox):
    """Custom ViewBox to provide axis‑specific mouse‑wheel zooming with limits.

    * Scrolling inside the plot area zooms both X and Y.
    * Scrolling over the bottom axis label zooms X only.
    * Scrolling over the left axis label zooms Y only.
    * Zoom range is clamped to user‑defined bounds (set via ``update_bounds``).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_limits = None  # (min, max)
        self.y_limits = None  # (min, max)

    def update_bounds(self, x_min, x_max, y_min, y_max):
        """Update the permissible zoom bounds.
        The view will never go outside these ranges.
        """
        self.x_limits = (x_min, x_max)
        self.y_limits = (y_min, y_max)
        # Ensure current view respects new bounds
        self.enforce_limits()

    def enforce_limits(self):
        vr = self.state.get('viewRange', None)
        if not vr:
            return
        x_range, y_range = vr
        x0, x1 = x_range
        y0, y1 = y_range
        if self.x_limits:
            if self.x_limits[0] is not None:
                x0 = max(x0, self.x_limits[0])
            if self.x_limits[1] is not None:
                x1 = min(x1, self.x_limits[1])
        if self.y_limits:
            if self.y_limits[0] is not None:
                y0 = max(y0, self.y_limits[0])
            if self.y_limits[1] is not None:
                y1 = min(y1, self.y_limits[1])
        self.setXRange(x0, x1, padding=0)
        self.setYRange(y0, y1, padding=0)

    def get_plot_item(self):
        p = self.parentItem()
        while p is not None:
            if hasattr(p, 'getAxis'):
                return p
            p = p.parentItem()
        return None

    def wheelEvent(self, ev: QtCore.QEvent, axis=None):
        # Determine zoom factor based on wheel direction
        if hasattr(ev, 'angleDelta') and ev.angleDelta() is not None:
            delta = ev.angleDelta().y()
        else:
            delta = ev.delta()
        if delta == 0:
            ev.accept()
            return
        factor = 0.8 if delta > 0 else 1.25  # wheel up = zoom in
        # Identify cursor location relative to the ViewBox bounding rect
        scene_pos = ev.scenePos()
        vb_rect = self.sceneBoundingRect()
        x = scene_pos.x()
        y = scene_pos.y()

        margin = 30  # Pixels of margin to extend axis detection area
        # Check axis using parameter or bounding rect logic
        is_bottom = (axis == 0) or ((y > vb_rect.bottom()) and (vb_rect.left() - margin <= x <= vb_rect.right() + margin))
        is_left = (axis == 1) or ((x < vb_rect.left()) and (vb_rect.top() - margin <= y <= vb_rect.bottom() + margin))
        # Current view ranges
        x_range, y_range = self.state['viewRange']
        x0, x1 = x_range
        y0, y1 = y_range
        # Get mouse position in view (data) coordinates
        mouse_in_view = self.mapSceneToView(scene_pos)
        mx = mouse_in_view.x()
        my = mouse_in_view.y()

        if is_bottom:
            # Zoom X only centered at mouse X
            new_x0 = mx - (mx - x0) * factor
            new_x1 = mx + (x1 - mx) * factor
            new_y0, new_y1 = y0, y1
        elif is_left:
            # Zoom Y only centered at mouse Y
            new_y0 = my - (my - y0) * factor
            new_y1 = my + (y1 - my) * factor
            new_x0, new_x1 = x0, x1
        else:
            # Zoom both axes centered at mouse position
            new_x0 = mx - (mx - x0) * factor
            new_x1 = mx + (x1 - mx) * factor
            new_y0 = my - (my - y0) * factor
            new_y1 = my + (y1 - my) * factor
        # Apply limits if defined
        if self.x_limits:
            if self.x_limits[0] is not None:
                new_x0 = max(new_x0, self.x_limits[0])
            if self.x_limits[1] is not None:
                new_x1 = min(new_x1, self.x_limits[1])
        if self.y_limits:
            if self.y_limits[0] is not None:
                new_y0 = max(new_y0, self.y_limits[0])
            if self.y_limits[1] is not None:
                new_y1 = min(new_y1, self.y_limits[1])
        # Update view
        self.setXRange(new_x0, new_x1, padding=0)
        self.setYRange(new_y0, new_y1, padding=0)
        ev.accept()
