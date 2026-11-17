"""Desktop GUI that takes a cohort of wide-field recordings from raw TIFFs to results.

Spec: ``gui_spec.md`` at the repository root. Run with ``python -m wfci_gui``.

Layers (GUI_DESIGN_GUIDELINES.md section 2):
  * ``wfci_gui.logic``  -- pure Python, no Qt; cohort files, tables, commands, status.
  * ``wfci_gui.controller`` -- the only code that changes application state.
  * ``wfci_gui.views`` / ``wfci_gui.window`` -- Qt widgets that redraw from state.

The GUI never imports the analysis code (``wfci`` or the study scripts) into its own
process: every step runs as a ``conda run`` subprocess (gui_spec.md D1).
"""

__version__ = "0.1.0"
