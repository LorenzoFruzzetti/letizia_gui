"""Desktop GUI that takes a cohort of wide-field recordings from raw TIFFs to results.

Run with ``python -m wfci_gui``.

Layers (GUI_DESIGN_GUIDELINES.md section 2):
  * ``wfci_gui.logic``  -- pure Python, no Qt; cohort files, tables, commands, status.
  * ``wfci_gui.controller`` -- the only code that changes application state.
  * ``wfci_gui.views`` / ``wfci_gui.window`` -- Qt widgets that redraw from state.

The GUI never imports the analysis code (``wfci`` or the study scripts) into its own
process: every step runs as a ``conda run`` subprocess.
"""

# The one place the application version is defined. A cohort file records the version
# that last saved it (``app_version``); the GUI refuses a file saved by a newer one.
APP_VERSION = "0.1.0"
__version__ = APP_VERSION
