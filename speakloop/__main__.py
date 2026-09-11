# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""``python -m speakloop`` support: a shim, with no logic of its own.

The entry point in pyproject.toml names ``speakloop.cli:main`` rather than
anything here on purpose. Under ``python -m speakloop`` this file is executed as
the module ``__main__``, so code living here would exist twice in a process
that also imported ``speakloop.__main__`` under its real name - two module
objects, two copies of everything at module level. Keeping the logic in
``cli.py`` makes that impossible instead of merely unlikely.

Why keep this form at all when the console script exists: it is the way in
when the script is not reachable. The interpreter's ``Scripts`` directory is
routinely missing from PATH on Windows, and a virtual environment moved to
another directory leaves script shims pointing at an interpreter that is no
longer there. ``python -m speakloop`` works in both cases.
"""

from speakloop.cli import main

# The guard costs nothing and closes the remaining way to start the application
# by accident: both supported forms (``python -m speakloop`` and
# ``python -m speakloop.__main__``) run this file AS ``__main__`` and are
# unaffected, while a plain ``import speakloop.__main__`` - a documentation tool
# walking the package, a stray editor auto-import - does not open a window.
if __name__ == "__main__":
    main()
