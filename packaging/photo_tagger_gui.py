# mypy: ignore-errors
"""
Entry point for the bundled Photo Tagger desktop app.

PyInstaller starts the .app here. Passing ``--selftest`` imports the heavy collaborators and exits,
so a freshly built bundle can be verified (every module actually got bundled) without opening a
window or needing a display.
"""

import os
import sys


if getattr(sys, "frozen", False):
    # In a PyInstaller bundle the logfire pydantic plugin (pulled in transitively by pydantic-ai)
    # calls inspect.getsource(), which has no .py source to read in a frozen app and raises. The
    # app does not use logfire, so disable pydantic plugins when running bundled.
    os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")


def main() -> int:
    """Launch the GUI, or run a quick import self-test for build verification."""
    if "--selftest" in sys.argv:
        # Import the third-party libs and the photo_tagger modules that pull them in, so a missing
        # bundled dependency surfaces here instead of when the user first clicks a button.
        import exiftool
        import PIL.Image
        import pydantic_ai
        import rawpy

        import photo_tagger.ai
        import photo_tagger.gui
        import photo_tagger.metadata
        import photo_tagger.pipeline

        sys.stdout.write("selftest ok\n")
        return 0

    from photo_tagger.gui import launch

    return launch()


if __name__ == "__main__":
    sys.exit(main())
