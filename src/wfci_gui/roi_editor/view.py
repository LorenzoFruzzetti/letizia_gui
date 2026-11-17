"""One image panel of the ROI editor: ``ParityPanel``.

The window shows two, titled only "odd" and "even" (no channel label, to avoid bias).
Each draws the same page model: the image, the boxes (``pg.RectROI``, whole-pixel snap,
four corner handles, no rotation, as in the old editor), the box labels, Bregma (a
draggable target), Lambda (draggable), the Bregma-Lambda axis and the channel rectangle
(``pg.ROI``, rotation locked to the axis, scale handles for length and both widths, as
in scripts/channel_rectangle_viewer.py).

A panel never edits anything: every user gesture is passed to ``events`` (the
controller), which changes the model and asks both panels to redraw. ``events.syncing``
is True while the code moves items, so the signals that a programmatic ``setPos`` emits
are not read back as user gestures.

Coordinates: the view is the 128 grid, row-major, pixel (r, c) covering [r, r+1) x
[c, c+1), y inverted so row 0 is at the top (see roi_editor.py, "Geometry, once").
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from qtkit.qt import QtCore

from roi_editor import pixel_bounds  # noqa: E402  (repo root, on sys.path via logic)

HANDLE_SIZE = 7          # corner-handle radius in screen px (the old editor's)
pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)


class ParityPanel(pg.PlotWidget):
    """One parity's image with every editable item of the page on top."""

    def __init__(self, parity: str, events) -> None:
        super().__init__(title=parity)
        self.parity = parity
        self.events = events
        self.setFocusPolicy(QtCore.Qt.NoFocus)        # keys go to the window's actions
        plot = self.getPlotItem()
        plot.invertY(True)
        plot.setAspectLocked(True)
        plot.showAxes(True, showValues=True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        plot.addItem(self.image_item)
        # Shown instead of an image when the parity has none (old single preview cached).
        self.note = pg.TextItem("", color=(200, 200, 200), anchor=(0.5, 0.5))
        plot.addItem(self.note)
        self.rois: dict[str, pg.RectROI] = {}
        self.labels: dict[str, pg.TextItem] = {}

        self.bregma = pg.TargetItem(pos=(0, 0), size=18, movable=True, pen=pg.mkPen("m", width=2),
                                    hoverPen=pg.mkPen("y", width=3), label=None)
        self.lambda_ = pg.TargetItem(pos=(0, 0), size=16, movable=True, symbol="x",
                                     pen=pg.mkPen("g", width=2), hoverPen=pg.mkPen("y", width=3), label=None)
        for target in (self.bregma, self.lambda_):
            target.setZValue(100)
            plot.addItem(target)
        self.bregma.sigPositionChanged.connect(lambda item: events.bregma_dragged(self, item.pos()))
        self.lambda_.sigPositionChanged.connect(lambda item: events.lambda_dragged(self, item.pos()))
        self.bregma.sigPositionChangeFinished.connect(lambda item: events.gesture_finished())
        self.lambda_.sigPositionChangeFinished.connect(lambda item: events.gesture_finished())
        self.axis_line = pg.PlotDataItem(pen=pg.mkPen("g", width=1, style=QtCore.Qt.DashLine))
        self.axis_line.setZValue(99)
        plot.addItem(self.axis_line)

        # Rotation locked to the axis: no rotate handle; a drag is projected back onto the
        # axis by the model (channel_rectangle.factors_from_roi).
        self.rectangle = pg.ROI((0, 0), (1, 1), pen=pg.mkPen((255, 140, 0), width=2), rotatable=False)
        self.rectangle.addScaleHandle([1, 0.5], [0, 0.5])      # length: end edge, start edge fixed
        self.rectangle.addScaleHandle([0.5, 1], [0.5, 0.5])    # width, symmetric about the axis
        self.rectangle.addScaleHandle([0.5, 0], [0.5, 0.5])
        self.rectangle.setZValue(98)
        self.rectangle.sigRegionChanged.connect(lambda roi: events.rectangle_dragged(self, roi, False))
        self.rectangle.sigRegionChangeFinished.connect(lambda roi: events.rectangle_dragged(self, roi, True))
        plot.addItem(self.rectangle)

        self.scene().sigMouseClicked.connect(self._clicked)

    # ----- images ----------------------------------------------------------------------
    def set_image(self, image: np.ndarray | None, note: str = "") -> None:
        if image is None:
            self.image_item.clear()
            self.note.setText(note)
            self.note.setVisible(True)
            return
        # Image before any transform; 1 px = 1 grid unit.
        self.image_item.setImage(image, autoLevels=False)
        self.note.setVisible(False)

    def set_levels(self, levels) -> None:
        if self.image_item.image is not None:
            self.image_item.setLevels(levels)

    def fit(self, shape: tuple[int, int]) -> None:
        rows, cols = shape
        self.getPlotItem().setRange(xRange=(0, cols), yRange=(0, rows), padding=0.02)
        self.note.setPos(cols / 2, rows / 2)

    # ----- boxes -----------------------------------------------------------------------
    def build_boxes(self, labels) -> None:
        """One RectROI + label per box (called when the set of labels changes)."""
        plot = self.getPlotItem()
        for item in [*self.rois.values(), *self.labels.values()]:
            plot.removeItem(item)
        self.rois.clear()
        self.labels.clear()
        for label in labels:
            roi = pg.RectROI(pos=(0, 0), size=(1, 1), pen=pg.mkPen("c", width=2), hoverPen=pg.mkPen("y", width=2),
                             # Whole-pixel edits only; no rotation (a box is an array slice).
                             translateSnap=True, scaleSnap=True, snapSize=1.0,
                             rotatable=False, resizable=True, removable=False)
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addScaleHandle([1, 0], [0, 1])
            roi.addScaleHandle([0, 1], [1, 0])
            for handle in roi.getHandles():
                handle.radius = HANDLE_SIZE
                handle.buildPath()
            roi.sigRegionChanged.connect(lambda item, lbl=label: self.events.box_dragged(self, lbl, item))
            roi.sigRegionChangeFinished.connect(lambda item: self.events.gesture_finished())
            roi.sigClicked.connect(lambda item, event, lbl=label: self.events.box_clicked(lbl))
            plot.addItem(roi)
            text = pg.TextItem(label, color="c", anchor=(0, 1))
            plot.addItem(text)
            self.rois[label], self.labels[label] = roi, text

    def sync(self, page, selected: str | None, bad: set[str], rectangle_state: dict | None,
             skip: str | None = None) -> None:
        """Put every item where the page says it is. ``skip``: a box being dragged here."""
        for label, box in page.boxes.items():
            r0, r1, c0, c1 = pixel_bounds(box, page.y_1, page.x_2)
            roi = self.rois[label]
            if label != skip:
                roi.setPos(c0, r0, finish=False)
                roi.setSize((c1 - c0 + 1, r1 - r0 + 1), finish=False)
            colour = "r" if label in bad else "y" if label == selected else "c"
            roi.setPen(pg.mkPen(colour, width=3 if label == selected else 2))
            self.labels[label].setColor(colour)
            self.labels[label].setPos(c0, r0)          # anchor (0, 1): just above the box
        bregma_x, bregma_y = page.x_2 - 0.5, page.y_1 - 0.5
        lambda_row, lambda_col = page.lambda_point
        self.bregma.setPos(bregma_x, bregma_y)
        self.lambda_.setPos(lambda_col - 0.5, lambda_row - 0.5)
        self.axis_line.setData([bregma_x, lambda_col - 0.5], [bregma_y, lambda_row - 0.5])
        self.rectangle.setVisible(rectangle_state is not None)
        if rectangle_state is not None:
            self.rectangle.setState({"pos": pg.Point(*rectangle_state["pos"]),
                                     "size": pg.Point(*rectangle_state["size"]),
                                     "angle": rectangle_state["angle"]})

    # ----- gestures ----------------------------------------------------------------------
    def _clicked(self, event) -> None:
        """Double-click: put Bregma on the clicked pixel."""
        if event.double():
            position = self.getPlotItem().vb.mapSceneToView(event.scenePos())
            self.events.double_clicked(position.y(), position.x())
