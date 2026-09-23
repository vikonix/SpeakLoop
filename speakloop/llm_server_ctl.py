# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Valeriy Kovalev

"""Lifecycle control for the local LLM server subprocess.

Used with the "llama-server" backend, the only one that runs a server of its
own: the controller launches the official llama.cpp binary, waits until it
answers, and terminates it on app shutdown. The dialogue itself goes through
LLMManager (llm.py).

The server speaks the same OpenAI-compatible API as LM Studio and answers 503
while the model is still loading, so the readiness poll is just
LLMManager.check_connection. The caller has already pointed that client at the
server (config.LLM_URL).

A server already listening on the port is used as it is rather than replaced:
an app that died without quit_app leaves one behind, and a second one could not
bind the port anyway (see _use_running_server).
"""

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from speakloop import bootstrap, config, llama_server_fetch
from speakloop.llm import LLMManager

# How long to wait for a graceful exit before killing the subprocess.
SERVER_TERMINATE_TIMEOUT_SEC = 5

# How long the port probe waits for a TCP connection. The port is on this
# machine and the answer comes from the kernel, so this is a guard against a
# blocked network stack and not a real wait.
PORT_PROBE_TIMEOUT_SEC = 0.5

# How long the /props read of an adopted server may take. That server has
# already answered /v1/models, so this is a guard against a hang and not a
# wait either.
PROPS_TIMEOUT_SEC = 2.0

# llama-server tuning that must never be left to the binary's own defaults:
#   --parallel 1     : the default (-1) opens several slots, splits the context
#                      between them and routes a request to whichever slot is
#                      most similar, which fragments the prefix cache.
#   --cache-reuse 256: keeps prefix reuse at the level llama-cpp-python gave,
#                      which the stable system prompt of llm.py relies on.
# --ctx-size is passed explicitly for the same reason: the default (0) takes
# the model's own training context and inflates the KV cache to fill free VRAM.
LLAMA_SERVER_PARALLEL_SLOTS = 1
LLAMA_SERVER_CACHE_REUSE = 256

# The --n-gpu-layers value that passes no argument at all (see
# config.GPU_LAYERS_WORDS).
AUTO_GPU_LAYERS = "auto"

# Smallest prompt batch that ggml computes on the GPU when some layers stay on
# the CPU ("op offload"; ggml's own default is 32). llama-server holds back the
# last few tokens of a prompt, so a learner's reply of about 34 tokens arrives
# as a batch of 30 and ran on the CPU: 2.5 s instead of 1.5 s on the reference
# laptop. At 16 tokens both paths cost the same, so a lower value gains
# nothing (docs/model-parameters.md, section 4.9). No effect when every layer
# is on the GPU or on a CPU build.
OP_OFFLOAD_MIN_BATCH_VAR = "GGML_OP_OFFLOAD_MIN_BATCH"
OP_OFFLOAD_MIN_BATCH = 16


def server_environment() -> dict:
    """Environment of the llama-server subprocess.

    A copy of this process's environment, so the server still finds its
    CUDA runtime on PATH. The batch threshold is set with setdefault: a value
    the owner exported before the start wins, which is how another threshold
    can be tried without a code change.
    """
    environment = dict(os.environ)
    environment.setdefault(OP_OFFLOAD_MIN_BATCH_VAR, str(OP_OFFLOAD_MIN_BATCH))
    return environment


def find_llama_server(setting: str) -> str:
    """The llama-server binary to launch, or "" when there is none.

    *setting* is config.LLAMA_SERVER_PATH and wins when it is set. Otherwise
    the binary llama_server_fetch installed, then one on PATH. Called at every
    start: the binary can be installed or removed while the app is closed.
    """
    if setting:
        return setting
    installed = llama_server_fetch.installed_exe()
    if installed is not None:
        return str(installed)
    return shutil.which("llama-server") or ""


def llama_server_command(exe_path: str, model_path: str, host: str, port: int,
                         n_gpu_layers: str, n_ctx: int, api_key: str) -> list:
    """Command line for the llama.cpp binary (the "llama-server" backend).

    GPU layers: with *n_gpu_layers* "auto" no --n-gpu-layers is passed, so
    llama.cpp fits the layers into the free VRAM itself (-fit, on by
    default). Any other value is passed as it is, and an explicit value
    switches that fit off (docs/model-parameters.md, section 3).

    -fitc equals the context: without it the fit shrinks the context, with no
    error, when the VRAM is short. The lesson needs its context more than a few
    GPU layers, so the fit has to give up layers instead. Passed with a manual
    layer count too, where it has no effect, so the command has one shape.

    --no-ui drops the bundled browser UI: the app talks HTTP only, and not
    serving the assets keeps the surface small. (--no-webui is the same switch
    under its former name; the binary now reports that spelling as deprecated.)

    --api-key answers the warning llama-server prints when it starts without
    one. The server allows every CORS origin, so without a key any page open in
    a browser could call this port and read the answer. The value is the one
    LLMManager already sends, so requiring it costs nothing.
    """
    command = [
        exe_path,
        "-m", model_path,
        "--host", host,
        "--port", str(port),
    ]
    if n_gpu_layers != AUTO_GPU_LAYERS:
        command += ["--n-gpu-layers", str(n_gpu_layers)]
    command += [
        "--ctx-size", str(n_ctx),
        "-fitc", str(n_ctx),
        "--parallel", str(LLAMA_SERVER_PARALLEL_SLOTS),
        "--cache-reuse", str(LLAMA_SERVER_CACHE_REUSE),
        "--api-key", api_key,
        "--no-ui",
    ]
    return command


def port_is_busy(host: str, port: int) -> bool:
    """True when something already listens on *host:port*.

    Asked before every launch, because llama-server cannot bind a port that is
    taken and exits at once when it tries. A plain TCP connection is the whole
    test: it sends no HTTP request, answers in microseconds on a local port,
    and reports a port held by a program that does not speak the API just as
    well as one held by a server that does.
    """
    try:
        with socket.create_connection((host, port), PORT_PROBE_TIMEOUT_SEC):
            return True
    except OSError:
        # Refused, unreachable or timed out: nothing is listening there.
        return False


def server_properties(host: str, port: int, api_key: str) -> Optional[dict]:
    """The /props answer of a running llama-server, or None.

    Not asked through the OpenAI client: /props is llama.cpp's own endpoint and
    sits outside the /v1 prefix that client is built around, so a plain GET is
    shorter and does not depend on the library's internals. The key is sent
    because the server is started with --api-key and answers 401 without it.

    None for every failure, including an answer that is not a JSON object:
    the only caller is diagnostic, and a server that answers must never be
    refused over what this function could not read.
    """
    url = f"http://{host}:{port}/props"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request,
                                    timeout=PROPS_TIMEOUT_SEC) as answer:
            payload = json.load(answer)
    except (OSError, ValueError) as exc:
        # OSError covers urllib's URLError and HTTPError (another server has no
        # /props at all), ValueError a body that is not JSON.
        logging.info("Could not read %s: %s", url, exc)
        return None
    return payload if isinstance(payload, dict) else None


def log_compute_devices(exe_path: str) -> None:
    """Record which devices llama-server sees; warn on a silent CPU fallback.

    llama.cpp drops to the CPU *without an error* when a CUDA build cannot load
    its runtime DLLs: it still logs "offloaded N/N layers to GPU", still
    answers every request, and is merely about three times slower. At the
    default verbosity the server's own log shows neither the buffer names nor
    the chosen devices, so nothing in a normal run would reveal this - only a
    speed comparison would, and only if someone thought to make one.

    Purely diagnostic: the probe loads no model, and any failure here is
    reported and shrugged off rather than blocking a server that would have
    started fine.
    """
    exe = Path(exe_path)
    try:
        devices = llama_server_fetch.list_devices(exe).strip()
    except llama_server_fetch.LlamaServerFetchError as exc:
        logging.warning("Could not list llama-server compute devices: %s", exc)
        return
    logging.info("llama-server compute devices:\n%s", devices)

    variant_name = llama_server_fetch.installed_variant(exe)
    if variant_name is None:
        # A binary the user manages themselves: we have no idea which backend
        # it was built with, so the listing above is the whole report.
        return
    pattern = llama_server_fetch.VARIANTS[variant_name].device_pattern
    if pattern is None or re.search(pattern, devices):
        return
    logging.warning(
        "llama-server is installed as %s but reports no matching compute "
        "device - it will run on the CPU, roughly three times slower, without "
        "reporting an error. Re-run `python -m speakloop.llama_server_fetch "
        "--force` to repair the installation.", variant_name)


class LLMServerController:
    """Starts and stops the llama-server subprocess."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._log_file = None
        # Serializes subprocess creation (start) against teardown (shutdown)
        # and makes shutdown() itself safe to reach from two threads at once
        # (loader thread on a start() failure path + Tk thread in quit_app).
        self._shutdown_lock = threading.Lock()
        # Set by shutdown(). A start() that loses the race to a quit_app
        # shutdown must not spawn a server afterwards - nothing would ever
        # terminate it. One-way by design: start() runs once per process
        # (load_components) and is never retried after shutdown.
        self._shutdown_requested = False
        # Set when start() found a server already listening and used it as it
        # is. Public, because app.py has to tell the user: that server keeps
        # the model, context size and GPU layers it was started with, which are
        # not necessarily the ones this run configured.
        self.adopted = False
        # One short sentence for the window after a failed start(), or None.
        # The full reason is always in the log; this is the part of it a user
        # can act on (see app.py, which shows it in the chat).
        self.last_error: Optional[str] = None
        # Per-slot context the ready server reports through /props, or None
        # when it did not say. Read for a launched server as well as for an
        # adopted one: -fitc should keep the configured value, and app.py
        # tells the user when the server still has less.
        self.served_n_ctx: Optional[int] = None

    def _build_command(self) -> Optional[list]:
        """Command line for llama-server, or None on a bad setup.

        Every "cannot start" reason is logged here and reported to the caller
        as None, so start() has a single failure path and app.py keeps its
        one error message for the user. Each reason also leaves a short
        sentence in last_error, which is what that message says.
        """
        model_path = config.EXTERNAL_MODEL_PATH
        if not model_path:
            logging.error("EXTERNAL_MODEL_PATH is empty - cannot start the LLM server.")
            self.last_error = "No GGUF model is configured."
            return None

        exe_path = find_llama_server(config.LLAMA_SERVER_PATH)
        if not exe_path:
            logging.error(
                "llama-server binary not found: settings.json "
                "'llama_server_path' is empty, %s holds no installation and "
                "no llama-server is on PATH. Run "
                "`python -m speakloop.llama_server_fetch` or set the path.",
                llama_server_fetch.INSTALL_DIR)
            self.last_error = ("The llama-server binary was not found. Run "
                               "`python -m speakloop.llama_server_fetch`.")
            return None
        if not os.path.isfile(exe_path):
            logging.error("llama-server binary not found at %s (settings.json "
                          "'llama_server_path').", exe_path)
            self.last_error = (f"There is no llama-server binary at "
                               f"{exe_path}.")
            return None
        return llama_server_command(
            exe_path, model_path, config.LLM_SERVER_HOST,
            config.LLM_SERVER_PORT, config.EXTERNAL_N_GPU_LAYERS,
            config.EXTERNAL_N_CTX, config.LLM_SERVER_API_KEY)

    def _use_running_server(self, llm_mgr: LLMManager) -> bool:
        """Use a server that already listens on the configured port.

        This is what makes an abnormal exit survivable. Nothing terminates the
        subprocess when the app dies without quit_app (a crash, "stop" in an
        IDE, taskkill), so the server keeps the port and the VRAM. A launch
        would then fail to bind and report a broken setup, although a working
        server is right there - so that server is used instead, and is left
        running on exit exactly as it was found.

        Returns True when such a server answered. Returns False with a
        last_error when the port is held by something that does not answer the
        API, because a launch cannot have that port either. A free port is
        neither case: False with no error, and start() goes on to launch.
        """
        host, port = config.LLM_SERVER_HOST, config.LLM_SERVER_PORT
        if not port_is_busy(host, port):
            return False

        logging.info("Port %s:%s is already in use - asking what is there.",
                     host, port)
        if not llm_mgr.check_connection(silent=True):
            logging.error(
                "Port %s:%s is in use by something that does not answer the "
                "OpenAI API. A new llama-server cannot bind that port, so no "
                "server is started.", host, port)
            self.last_error = (
                f"Port {port} is in use by another program. Close it (an old "
                f"llama-server from a crashed session is the usual cause) and "
                f"start SpeakLoop again.")
            return False

        self.adopted = True
        logging.warning(
            "Using the llama-server that already listens on %s:%s. Its model, "
            "context size and GPU layers are the ones it was started with, "
            "not the ones this run configured.", host, port)
        self._log_served_model(llm_mgr)
        self.served_n_ctx = self._log_served_context(host, port)
        return True

    @staticmethod
    def _log_served_model(llm_mgr: LLMManager) -> None:
        """Record which model the adopted server serves; warn on a mismatch.

        Diagnostics only, and silent about its own failures: the server has
        already answered, so nothing here may turn a usable server into a
        failed start.

        llama-server reports the GGUF file it loaded as the model id, which is
        what makes the comparison possible at all. Another OpenAI-compatible
        server may report a name of its own, and the mismatch is then a line in
        the log for whoever reads it after a strange lesson - not a refusal,
        because the name says nothing about whether the server works.
        """
        try:
            served = [model.id for model in llm_mgr.client.models.list().data]
        except Exception as exc:
            logging.info("Could not read the model list of the running "
                         "server: %s", exc)
            return
        logging.info("The running server reports these models: %s",
                     ", ".join(served) or "(none)")
        expected = os.path.basename(config.EXTERNAL_MODEL_PATH or "")
        if expected and not any(expected in name for name in served):
            logging.warning(
                "The running server does not report %s, the model this run "
                "configured. The lesson runs on whatever that server has "
                "loaded.", expected)

    @staticmethod
    def _log_served_context(host: str, port: int) -> Optional[int]:
        """Record the context size of the ready server; warn if it is small.

        Returns that size, or None when the server did not report one.

        The number that matters most about a server this run did not start: a
        conversation sized for a larger context is truncated by a smaller one
        silently, on the server, and nothing else in the app would report it.
        A server this run started is asked too, because llama.cpp's memory fit
        can shrink the context without an error, and -fitc is the only guard
        against that. A larger context than configured costs only VRAM, so
        only the harmful direction is a warning.

        The value read is the per-slot context (`n_ctx` of
        default_generation_settings), which is the one a request actually gets.
        With --parallel 1 it equals --ctx-size, and a foreign server started
        with several slots is exactly the case where the two differ.

        Diagnostic like _log_served_model: a server that answers is never
        refused over this.
        """
        props = server_properties(host, port, config.LLM_SERVER_API_KEY)
        if props is None:
            return None
        settings = props.get("default_generation_settings")
        n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        # bool is a subclass of int, and a JSON true is not a context size.
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool):
            logging.info("The running server did not report a context size.")
            return None
        logging.info("The running server has a context size of %s tokens; "
                     "this run is configured for %s.",
                     n_ctx, config.EXTERNAL_N_CTX)
        if n_ctx < config.EXTERNAL_N_CTX:
            logging.warning(
                "The running server has a smaller context (%s tokens) than "
                "this run configured (%s). A long conversation is cut by that "
                "server, not by SpeakLoop.", n_ctx, config.EXTERNAL_N_CTX)
        return n_ctx

    def start(self, llm_mgr: LLMManager) -> bool:
        """Launch the server subprocess and block until it responds.

        Readiness is probed through ``llm_mgr``, whose client must already
        point at the local server (config.LLM_URL). Returns False on a busy port (see _use_running_server), an
        unusable configuration (see _build_command), an early subprocess exit,
        a startup timeout, or when shutdown() has already been requested.
        Every False except the last leaves its reason in last_error.

        Nothing is launched when a server already answers on the port: True
        then means "a server is ready", not "this controller owns one".
        """
        self.adopted = False
        self.last_error = None
        self.served_n_ctx = None
        if self._use_running_server(llm_mgr):
            return True
        if self.last_error is not None:
            # The port is taken by something that does not answer the API: a
            # launch would only fail to bind it, so there is nothing to try.
            return False

        cmd = self._build_command()
        if cmd is None:
            return False

        # Before the model load makes the wait long: the probe is sub-second
        # and its answer is the only record of which backend came up.
        log_compute_devices(cmd[0])

        log_path = config.LLM_SERVER_LOG_FILE
        environment = server_environment()
        logging.info(f"Starting LLM server: {' '.join(cmd)}")
        # Logged because it is not on the command line, and a run with an
        # owner-exported value is otherwise impossible to tell apart.
        logging.info("LLM server environment: %s=%s", OP_OFFLOAD_MIN_BATCH_VAR,
                     environment[OP_OFFLOAD_MIN_BATCH_VAR])
        logging.info(f"LLM server output -> {log_path}")
        # Creation runs under the same lock as shutdown(), so the two cannot
        # interleave: either shutdown() runs first and the flag stops the
        # launch, or the subprocess is fully published before shutdown() gets
        # the lock and terminates it. Without this, a quit during startup
        # could leave a freshly spawned server orphaned.
        with self._shutdown_lock:
            if self._shutdown_requested:
                logging.info("LLM server start aborted: shutdown requested.")
                return False
            # Mode from bootstrap, not a literal "w": after an in-session
            # restart main.log is continued, and a server log that starts
            # over would cover only the second half of a session whose app
            # log covers all of it.
            log_mode = bootstrap.log_file_mode()
            self._log_file = open(log_path, log_mode,
                                  encoding="utf-8", buffering=1)
            if log_mode == "a":
                # The same seam marker main.log gets, in the file's own terms.
                # This one carries no timestamps of its own (it is the server's
                # raw stdout), so without a line here the two runs simply abut.
                self._log_file.write(
                    "\n----- log continues here: server restarted in-session "
                    "-----\n")
            try:
                self._process = subprocess.Popen(
                    cmd, stdout=self._log_file, stderr=self._log_file,
                    env=environment)
            except Exception:
                # Don't leak the just-opened log file when the launch itself
                # fails (e.g. a binary that is not executable); the exception
                # still propagates to the caller's error handling.
                self._log_file.close()
                self._log_file = None
                raise

        deadline = time.time() + config.LLM_SERVER_STARTUP_TIMEOUT
        while time.time() < deadline:
            # Snapshot the process reference: shutdown() (called from quit_app
            # on the Tk main thread while this loop runs on the loader thread)
            # sets self._process to None, and reading it twice would race that
            # and crash on None.poll(). A cleared reference means the app is
            # quitting - stop waiting quietly.
            process = self._process
            if process is None:
                logging.info("LLM server startup aborted: shutdown requested.")
                return False
            if process.poll() is not None:
                logging.error(f"LLM server exited unexpectedly (code {process.returncode}).")
                self.last_error = (
                    f"llama-server stopped at once (exit code "
                    f"{process.returncode}).")
                self.shutdown()  # nothing to terminate; closes the log file
                return False
            if llm_mgr.check_connection(silent=True):
                logging.info("LLM server is ready.")
                self.served_n_ctx = self._log_served_context(
                    config.LLM_SERVER_HOST, config.LLM_SERVER_PORT)
                return True
            time.sleep(1.0)

        logging.error("LLM server did not become ready within the timeout.")
        self.last_error = (
            f"llama-server did not answer within "
            f"{config.LLM_SERVER_STARTUP_TIMEOUT} seconds.")
        # The subprocess may still be loading the model - terminate it now
        # instead of leaving it holding VRAM until the app exits.
        self.shutdown()
        return False

    def shutdown(self):
        """Terminate the subprocess (kill on timeout) and close its log file.

        Safe to call repeatedly and when the server was never started - every
        step is a no-op then, which is also what leaves a server adopted by
        _use_running_server running: it is not this process's subprocess, and a
        server that outlived one app must outlive the next one too. Also called
        by start() on its failure paths, so
        it can run concurrently on the loader thread and the Tk main thread
        (quit_app); the lock makes the check-then-use on the process and log
        file atomic - the loser of the race sees None and does nothing. Also
        flags the controller so a start() still ahead of its Popen call aborts
        instead of spawning a server nothing would terminate.
        """
        with self._shutdown_lock:
            self._shutdown_requested = True
            process, self._process = self._process, None
            log_file, self._log_file = self._log_file, None

        if process is not None:
            if process.poll() is None:
                logging.info("Terminating LLM server subprocess...")
                process.terminate()
                try:
                    process.wait(timeout=SERVER_TERMINATE_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    logging.warning("LLM server did not exit cleanly - killing it.")
                    process.kill()
                    process.wait()  # reap the killed process (avoids a zombie on POSIX)

        if log_file is not None:
            log_file.close()
