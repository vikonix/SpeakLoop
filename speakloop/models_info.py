# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Identity and download size of every model SpeakLoop fetches.

The single place a repo id is written down: config, the fetchers and the
speech modules bind to these records rather than restating the strings, so a
download and the load that follows it cannot go to different repos.

**Stdlib only, no side effects.** The fetchers must never import
speakloop/config.py (config flips HF_HUB_OFFLINE=1 once the models are cached,
switching the network off precisely when a download is wanted), so facts they
share with config have to live in a module that depends on neither side.
install.py reads it too, before the requirements step has run.

Deliberately NOT here
---------------------
* **The llama-server binary.** Its size belongs next to its name and sha256 in
  speakloop/llama_server_fetch.py, so bumping the pinned release rewrites all
  three together; a stale size elsewhere would silently mis-state the download
  rather than fail. It is also not a model.
* **Cache paths.** They stay in speakloop/model_fetch.py, next to the code that
  writes into them.
* **Which models a run needs**: that depends on the active language and
  llm_backend, so it is computed at startup rather than being a property of a
  model.
* **The spaCy pipeline Kokoro's English G2P loads.** misaki downloads it with
  pip at the first English synthesis; nothing in SpeakLoop fetches it.
"""

from __future__ import annotations

from typing import NamedTuple

# ---------------------------------------------------------------------------
# What size_mb means
# ---------------------------------------------------------------------------
#
# BYTES OVER THE NETWORK, in decimal MB (bytes / 1_000_000). The installer
# promises traffic ("download X MB") to somebody who may be on a metered
# connection, so traffic is what is stored.
#
# It is NOT disk usage, and the two diverge by up to a factor of two: unpacked
# archives are larger, and the HF cache on Windows without symlink privileges
# COPIES files into snapshots/ instead of linking them
# (model_fetch._configure_symlink_fallback). A free-space warning must compute
# its own numbers.
#
# Decimal MB, not MiB, because that is how download sizes are advertised.
# llama_server_fetch._human() renders live progress under the same "MB" label
# and divides by 1_000_000 to match; dividing by 1024**2 makes a finished
# download report a number smaller than the one the user agreed to.

# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


class HfRepo(NamedTuple):
    """A Hugging Face repository fetched whole into the hub cache.

    label is display text (logs, the installer's prompts) and carries no size
    of its own: the number lives in size_mb and is formatted where it is shown,
    so the two can never contradict each other.
    """

    repo_id: str
    label: str
    size_mb: int


class HfFile(NamedTuple):
    """A single file pulled out of a Hugging Face repository.

    Separate from HfRepo because the size of one file is not the size of its
    repo, and because the download goes through hf_hub_download into a plain
    directory rather than through snapshot_download into the cache layout.
    """

    repo_id: str
    filename: str
    label: str
    size_mb: int


class PackagedModel(NamedTuple):
    """A model some other package downloads into its own cache directory.

    name is what that package calls the model, not a Hugging Face repo id: it
    is passed to the package's own loader, which resolves it however it likes.
    """

    name: str
    label: str
    size_mb: int


# ---------------------------------------------------------------------------
# Hugging Face hub repositories
# ---------------------------------------------------------------------------
#
# Re-snap the sizes with `python tools/measure_model_sizes.py` and commit the
# result whenever a repo id changes: a one-off per pin, never a runtime lookup.
# Measuring at startup would put a network round-trip in front of every
# install prompt, and eyeballed numbers come out optimistic by 5-10%.
#
# Measure the WHOLE snapshot, not the weights the app loads: snapshot_download
# without allow_patterns fetches every file in the repo.

# The repo faster-whisper resolves the model name "small" to. The app must load
# it by this repo id (or by the name that maps to it), or the load goes to a
# repo the installer never fetched.
WHISPER_SMALL = HfRepo(
    "Systran/faster-whisper-small",
    "faster-whisper small (speech recognition)",
    size_mb=486,  # measured 2026-09-11
)

KOKORO = HfRepo(
    "hexgrad/Kokoro-82M",
    "Kokoro-82M (text-to-speech, English)",
    size_mb=363,  # measured 2026-07-28
)

# Every hub repo, in the order model_fetch downloads them. Supertonic is NOT in
# this tuple: it does not use the hub cache (see below), so caching its repo
# under HF_HOME/hub would be dead weight the app never reads.
HF_REPOS: tuple[HfRepo, ...] = (
    WHISPER_SMALL,
    KOKORO,
)

# ---------------------------------------------------------------------------
# Models that live outside the hub cache
# ---------------------------------------------------------------------------

# Supertonic keeps its weights in its own directory: the package downloads them
# with snapshot_download(local_dir=...) into the directory named by the
# SUPERTONIC_CACHE_DIR env var. The weights are OpenRAIL-M licensed (the code is
# MIT), which is why they are downloaded rather than shipped with SpeakLoop.
SUPERTONIC = PackagedModel(
    "supertonic-3",
    "Supertonic 3 TTS (Spanish; weights OpenRAIL-M)",
    # Measured on disk rather than over the wire: which files the supertonic
    # package pulls out of Supertone/supertonic-3 is its own business, so the
    # repo total would be an upper bound. The cache directory IS the download -
    # the package writes with local_dir=, skipping the hub's blob/symlink
    # duplication.
    size_mb=404,  # measured 2026-07-28
)

# The GGUF chat model the llama-server backend loads. The filename matches the
# file the application loads from models/, so the app finds it without a
# settings change.
GGUF_CHAT = HfFile(
    "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
    "llama-3.2-3b-instruct-q4_k_m.gguf",
    "Llama 3.2 3B Instruct Q4_K_M (chat model for llama-server)",
    size_mb=2019,  # measured 2026-07-28
)
