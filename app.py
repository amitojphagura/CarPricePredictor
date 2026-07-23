"""
Convenience entrypoint: `python app.py` runs the Flask app defined in
app/main.py.

The actual app lives in app/main.py (not here) because this project already
has an `app/` package alongside this file -- `app.py` and `app/` can't both
be imported under the name `app` at once, so the real implementation had to
move into the package. Running this file directly still works since it's
executed as a script, not imported.

For containerization, point your WSGI server at `app.main:app`
(e.g. `gunicorn app.main:app`), not `app:app`.
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("app.main", run_name="__main__")
