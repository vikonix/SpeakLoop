# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""SpeakLoop application package: a local voice dialogue trainer.

The package is being built step by step next to the old root modules
(main.py, config.py, stt.py, llm.py, tts.py, llm_server/), which still run the
application until they move into this package.
"""

# Single source of truth for the application version (SemVer MAJOR.MINOR.PATCH,
# with an optional PEP 440 pre-release suffix while v1 is being built).
# pyproject.toml reads this value dynamically; runtime code imports it from here.
__version__ = "1.0.0.dev0"
