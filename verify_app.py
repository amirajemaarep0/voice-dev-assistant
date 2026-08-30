"""Execute app.py top-to-bottom outside Streamlit's runtime.

Widgets return their defaults and nothing renders, but any real error -
a NameError, a bad argument to a Streamlit call, a broken import - raises
here. Catches the class of bug you would otherwise only see by clicking
through the UI.
"""
import runpy
import warnings

warnings.filterwarnings("ignore")

try:
    runpy.run_path("app.py", run_name="__main__")
except Exception as exc:
    print(f"FAIL {type(exc).__name__}: {exc}")
    raise SystemExit(1)
print("APP_OK - app.py executed with no errors")
