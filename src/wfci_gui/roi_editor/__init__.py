"""ROI editor with odd | even panels, in a qtkit shell window.

Run with ``python -m wfci_gui.roi_editor`` (see ``__main__.py``). ``model.py`` and
``sources.py`` are Qt-free: the model re-implements the old editor's edits with the same
``roi_editor`` functions (``roi_editor.py`` is not modified), so a set saved here is
byte-identical to one saved there for the same edits (tests/test_roi_editor_model.py).
"""
