import matplotlib

# Use a non-interactive backend during tests.

# This allows Matplotlib to save figures without Tkinter or a GUI window.

matplotlib.use("Agg", force=True)
