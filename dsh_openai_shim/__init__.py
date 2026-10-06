"""Core of dsh-openai-shim: a tiny stdlib-only OpenAI-compatible proxy.

It sits between an OpenAI-compatible client (dsh / DeepSeek Harness) and an
upstream OpenAI endpoint (Socrates / LiteLLM / any sglang/vLLM endpoint),
forwarding requests while applying two safe rewrites:

  * ``reasoning_effort``  — remap to a supported value or drop it
  * ``max_tokens``        — clamp so input+output stays under the context cap
  * ``GET /v1/models``    — expose each model's context window under
                           ``context_window`` so the client's own context
                           detection reads the upstream's real limit

No third-party dependencies. Importable for unit testing; runnable standalone.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

__version__ = "0.2.7"

__all__ = ["__version__", "ShimConfig", "apply_rewrites", "make_handler", "serve",
           "ensure_shim", "shim_base_url", "is_shim_running", "DEFAULT_SHIM_PORT"]

from .runtime import (  # noqa: F401  (re-export)
    DEFAULT_SHIM_PORT,
    ensure_shim,
    is_shim_running,
    shim_base_url,
    shim_pidfile,
    stop_shim,
)


class ShimConfig:
    """Configuration for a single shim instance."""

    DEFAULT_EFFORT_MAP: dict[str, str] = {"high": "medium", "xhigh": "xhigh"}

    def __init__(
        self,
        upstream: str,
        listen_host: str = "127.0.0.1",
        listen_port: int = 8090,
        effort_mode: str = "map",          # "map" | "drop" | "off" | "low" | "none"
        effort_map: Optional[dict[str, str]] = None,
        token_cap: int = 100000,          # clamp completion max_tokens to this
        context_window: int = 0,         # upstream max_model_len (input+output); 0 = unknown
        upstream_key: Optional[str] = None,  # force a specific upstream key
        provider_name: Optional[str] = None, # proxy.conf provider this serves (for self-heal)
        timeout: int = 300,
    ) -> None:
        self.upstream = upstream.rstrip("/")
        self.listen_host = listen_host
        self.listen_port = int(listen_port)
        if effort_mode not in ("map", "drop", "off", "low", "none"):
            raise ValueError(f"effort_mode must be map|drop|off|low|none, got {effort_mode!r}")
        self.effort_mode = effort_mode
        self.effort_map = dict(effort_map or self.DEFAULT_EFFORT_MAP)
        self.token_cap = int(token_cap)
        self.context_window = int(context_window)
        self.upstream_key = upstream_key
        self.provider_name = provider_name
        self.timeout = int(timeout)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"ShimConfig(upstream={self.upstream!r}, port={self.listen_port}, "
            f"effort={self.effort_mode}, token_cap={self.token_cap})"
        )


def apply_rewrites(body: Optional[bytes], cfg: ShimConfig,
                   force_max_tokens: Optional[int] = None,
                   drop_thinking: bool = False,
                   drop_output_config: bool = False) -> tuple[bytes, list[str]]:
    """Apply the shim's rewrites to a request body.

    ``force_max_tokens`` (set only by the self-heal retry) pins ``max_tokens``
    to an exact value — the window minus the upstream's reported input count —
    overriding the usual cap/window clamp for that one attempt.

    ``drop_thinking`` (set only by the self-heal retry) removes the Anthropic
    ``thinking`` field entirely. Some backends reject extended-thinking
    requests (one requires ``budget_tokens`` when ``type`` is ``"enabled"``;
    another has no reasoning parser at all), so on that 400 the retry strips
    the field and resends a plain message.

    ``drop_output_config`` (set only by the self-heal retry) removes the
    nonstandard ``output_config`` field (added by dsh's ``pi-ai`` layer, e.g.
    ``{"effort": "high"}``). One backend 500s (opaque "Internal server error")
    whenever it is present — dropping it restores a clean 200 — so on a 500
    with the field in the request the retry strips it. Only fires when the
    request actually carries the field, so ordinary requests are untouched.

    Returns ``(new_body, changes)``. Bodies that are not JSON objects pass
    through untouched.
    """
    changes: list[str] = []
    if not body:
        return (body or b""), changes
    try:
        data: Any = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body, changes
    if not isinstance(data, dict):
        return body, changes

    # --- reasoning_effort ---------------------------------------------------
    if cfg.effort_mode != "off" and "reasoning_effort" in data:
        val = data["reasoning_effort"]
        if cfg.effort_mode == "drop":
            del data["reasoning_effort"]
            changes.append(f"dropped reasoning_effort={val!r}")
        elif cfg.effort_mode == "low":
            new = "low"
            data["reasoning_effort"] = new
            changes.append(f"!!!! reasoning_effort {val!r}->{new!r}")
        elif cfg.effort_mode == "none":
            new = "none"
            data["reasoning_effort"] = new
            changes.append(f"!!!! reasoning_effort {val!r}->{new!r}")
        else:
            new = cfg.effort_map.get(str(val))
            if new is not None and new != val:
                data["reasoning_effort"] = new
                changes.append(f"reasoning_effort {val!r}->{new!r}")

    # --- thinking drop (self-heal only) -----------------------------------
    # A backend that can't handle extended thinking 400s on the ``thinking``
    # field. On that error the retry strips it and resends a plain message —
    # the model still answers, just without a reasoning trace. Only fires when
    # a request actually carries the field, so ordinary requests are untouched.
    if drop_thinking and "thinking" in data:
        del data["thinking"]
        changes.append("dropped thinking (upstream rejected extended thinking)")

    # --- output_config drop (self-heal only) ------------------------------
    # A nonstandard field dsh's pi-ai layer adds (e.g. {"effort":"high"}); one
    # backend 500s on it. On that error the retry strips it and resends.
    if drop_output_config and "output_config" in data:
        del data["output_config"]
        changes.append("dropped output_config (upstream 500'd on it)")

    # --- max_tokens clamp ---------------------------------------------------
    # Keep the client's explicit max_tokens when it is already small (a short
    # answer is fine for a small cap); only OVERRIDE it when it would overflow
    # the model's context window (see _clamp_max_tokens), so a long context
    # never 400s — this is what the self-heal relies on to make the retry fit.
    data.pop("force_max_tokens", None)
    clamp_desc = _clamp_max_tokens(data, cfg, force=force_max_tokens)
    if clamp_desc:
        changes.append(clamp_desc)

    if not changes:
        return body, changes
    return (json.dumps(data).encode("utf-8"), changes)


def enrich_models_body(raw: Optional[bytes]) -> Optional[bytes]:
    """Rewrite a ``GET /v1/models`` body so the client's own context-window
    detection reads the upstream's REAL limit.

    OpenAI-compatible clients (dsh / DeepSeek Harness, hermes) discover a
    model's context window from ``/v1/models`` by a fixed set of field names.
    SGLang/vLLM report it as ``max_model_len`` — which dsh's field list does
    NOT include — so dsh can't read it and silently falls back to a hardcoded
    default (65536), even though the server's real window is smaller (e.g.
    64000). The first long turn then 400s: "…exceeds the model's maximum
    context length of 64000 tokens".

    We close that gap at the shim — the one hop EVERY provider goes through:
    for each model in ``data[]`` that reports a context-window field (under any
    name hermes-style discovery recognises) but not ``context_window`` (a field
    dsh DOES read), copy the discovered value into ``context_window``.

    Pure and idempotent: adds ``context_window`` once and never overwrites an
    existing one. Non-JSON, non-object, or non-``data``-list bodies, and bodies
    where no window field is discoverable, are returned unchanged (``None``) so
    the caller forwards the original bytes verbatim.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    models = data.get("data")
    if not isinstance(models, list):
        return None

    from .discover import _CONTEXT_KEYS, _cap, _extract_first_int

    changed = False
    for m in models:
        if not isinstance(m, dict) or "context_window" in m:
            continue  # absent -> nothing to add; present -> never clobber
        win = _cap(_extract_first_int(m, _CONTEXT_KEYS))
        if win is not None:
            m["context_window"] = win
            changed = True
    if not changed:
        return None
    return json.dumps(data).encode("utf-8")


def _estimate_input_tokens(data) -> int:
    """Rough input-token estimate from a chat request's FULL input.

    The server's "input tokens" is the model's entire prompt: the chat
    ``messages`` PLUS the ``system`` prompt PLUS every ``tool`` definition.
    dsh sends ~24 tools (thousands of tokens) — an estimate that counts only
    ``messages`` is far too small, which makes the proactive ``max_tokens``
    clamp too generous and overflows small-window backends (spark: 5068 real
    input vs ~145 estimated → ``input + max_tokens > window`` → 400). No
    tokenizer is available in a stdlib-only shim, so we count the serialized
    character length of each input field and divide by ~4 (an English token is
    ~3.5-4 chars). Overestimating is safe: it only LOWERS ``max_tokens`` (more
    conservative), never raises it.

    Accepts the whole request dict (counts ``messages`` + ``system`` +
    ``tools``); for backward compatibility a bare ``messages`` list is also
    accepted.
    """
    if isinstance(data, dict):
        parts = [data.get(k) for k in ("messages", "system", "tools")]
    elif isinstance(data, list):  # legacy caller passed just the messages list
        parts = [data]
    else:
        parts = []
    chars = 0
    for p in parts:
        if p is None:
            continue
        try:
            chars += len(json.dumps(p))
        except (ValueError, TypeError):
            chars += len(str(p))
    return max(1, chars // 4)


def _clamp_max_tokens(data: dict, cfg: "ShimConfig",
                      force: Optional[int] = None) -> Optional[str]:
    """Set ``data["max_tokens"]`` so ``input + output`` fits the model.

    The binding constraint is the *minimum* of the provider's output cap
    (``cfg.token_cap``) and the context window minus the estimated input and a
    small reserve (``cfg.context_window``) — because sglang/vLLM's
    ``max_model_len`` is the TOTAL window (input + output), not an output cap.
    ``force`` (the self-heal path) pins the value from the upstream's exact
    input count, overriding both. Returns a change description or ``None``.
    """
    if force is not None:
        forced = max(1, int(force))
        if data.get("max_tokens") != forced:
            prev = data.get("max_tokens")
            data["max_tokens"] = forced
            return (f"max_tokens {prev}->{forced} (context-fit, "
                    f"window={cfg.context_window})")
        return None
    cap = cfg.token_cap
    if not isinstance(cap, int) or cap <= 0:
        return None
    est = _estimate_input_tokens(data)
    eff = cap
    if isinstance(cfg.context_window, int) and cfg.context_window > 0:
        headroom = cfg.context_window - est - 128
        if headroom < cap:
            eff = max(1, headroom)
    prev = data.get("max_tokens")
    if isinstance(prev, int) and prev > eff:
        data["max_tokens"] = eff
        return f"max_tokens {prev}->{eff} (cap={cap}, window={cfg.context_window})"
    if "max_tokens" not in data and eff:
        data["max_tokens"] = eff
        return f"set max_tokens={eff} (cap={cap}, window={cfg.context_window})"
    return None


def _error_message(raw: bytes) -> str:
    """The human-readable ``message`` from an OpenAI-style error body, or ''."""
    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    err = payload.get("error")
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"]
    if isinstance(payload.get("message"), str):
        return payload["message"]
    return ""


def _parse_error_cap(raw: bytes) -> Optional[int]:
    """Completion-token ceiling reported in an upstream error body, or None."""
    message = _error_message(raw)
    if not message:
        return None
    from .discover import parse_completion_cap_from_error
    return parse_completion_cap_from_error(message)


def _parse_context_error(raw: bytes) -> tuple:
    """(window, input_tokens, total) from a context-window error body, or Nones."""
    message = _error_message(raw)
    if not message:
        return (None, None, None)
    from .discover import parse_context_window_error
    return parse_context_window_error(message)


def _is_thinking_error(raw: bytes) -> bool:
    """True when the upstream rejected the request because of the Anthropic
    ``thinking`` field.

    Two observed forms (both from strict backends, while lenient ones pass the
    same body through):
      * ``"thinking.budget_tokens is required when thinking.type is 'enabled'"``
        — extended thinking needs an explicit budget.
      * ``"Anthropic thinking is not supported for models without a
        reasoning parser"`` — the model can't do extended thinking at all.
    Matching on the ``thinking`` keyword in the message is enough: it is the
    field we would drop, so a retry without it is the correct response.
    """
    message = _error_message(raw)
    if not message:
        return False
    m = message.lower()
    if "thinking" not in m:
        return False
    # Narrow to errors that are actually about the thinking field, so an
    # unrelated error that merely mentions "thinking" in prose isn't retried.
    return any(
        k in m
        for k in (
            "budget_tokens",
            "budget",
            "reasoning parser",
            "thinking.type",
            "invalid_request",
            "unsupported",
            "not supported",
        )
    )


def _persist_provider_cap(name: Optional[str], cap, context_window=None) -> None:
    """Persist a self-healed cap / context window back to proxy.conf (best-effort).

    The in-memory ``cfg.token_cap`` / ``cfg.context_window`` are what fix the
    in-flight retry; this write makes them survive the next restart. Failure to
    write (no name, read-only home, …) must never break a request — the
    in-memory fix stands on its own, so swallow errors.
    """
    if not name:
        return
    try:
        from .config import persist_provider_cap
        if persist_provider_cap(name, cap, context_window):
            print(f"[shim] persisted token_cap={cap} context_window="
                  f"{context_window} for provider {name!r}", flush=True)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[shim] could not persist cap: {e}", flush=True)


# Hop-by-hop / re-settable headers that must not be forwarded verbatim.
_STRIP_REQ_HEADERS = {
    "host", "content-length", "accept-encoding", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "transfer-encoding", "upgrade",
}
_STRIP_RES_HEADERS = {
    "content-length", "connection", "keep-alive", "transfer-encoding",
    "upgrade", "proxy-authenticate", "proxy-authorization",
}


def _filter_headers(headers: Any, strip: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = list(headers.items())
    except AttributeError:
        items = [(k, v) for k, v in (headers or [])]  # http.client-style list
    for k, v in items:
        if k.lower() not in strip:
            out[k] = v
    return out


class _ClientGone(Exception):
    """The client hung up mid-response — stop streaming; this is NOT an upstream
    failure. Raised by the streaming write helper and caught by the stream loop."""


class _SSEMessageStartDedup:
    """Re-emit upstream SSE bytes event-by-event, dropping a duplicate
    ``message_start`` that repeats the first one's message id.

    LiteLLM's Anthropic ``/v1/messages`` adapter emits TWO ``message_start``
    events for a single message; dsh 0.2.0-rc.1's Anthropic client rejects the
    duplicate with ``MALFORMED_RESPONSE`` (``duplicate message_start``). A
    well-formed Messages stream has exactly ONE ``message_start``, so dropping
    the repeat is a no-op for the protocol — the stream is otherwise forwarded
    byte-identical.

    The class buffers raw bytes and returns COMPLETE events (terminated by a
    blank ``\\n\\n`` line) so the caller can forward each one as soon as it
    arrives, keeping the stream live. Only the trailing incomplete event is
    held back; ``flush()`` releases it at stream end.

    The filter is unconditional and safe for any stream: a normal OpenAI stream
    has no ``event: message_start`` lines (no-op), and a correct single-start
    Anthropic stream passes through unchanged.
    """

    __slots__ = ("_buf", "_seen", "_seen_id")

    def __init__(self) -> None:
        self._buf = b""
        self._seen = False            # a message_start has been forwarded
        self._seen_id: Optional[str] = None

    @staticmethod
    def _start_id(event: bytes) -> Optional[str]:
        for line in event.split(b"\n"):
            s = line.strip()
            if s.startswith(b"data:"):
                try:
                    obj = json.loads(s[5:].decode("utf-8", "replace"))
                except (ValueError, UnicodeDecodeError):
                    return None
                if isinstance(obj, dict):
                    m = obj.get("message")
                    if isinstance(m, dict) and isinstance(m.get("id"), str):
                        return m["id"]
        return None

    def _is_dup_start(self, event: bytes) -> bool:
        if not self._is_event_start(event) or not self._seen:
            return False
        mid = self._start_id(event)
        if self._seen_id is None or mid is None:
            return True               # can't confirm a distinct message — be safe
        return mid == self._seen_id

    @staticmethod
    def _is_event_start(event: bytes) -> bool:
        """True if this SSE event is a ``message_start``.

        Match the ``event: message_start`` LINE anywhere in the event, NOT
        position 0: dsh 0.2.0-rc.1's HTTP stack can deliver a stream whose
        FIRST bytes carry a framing artifact glued in front of the first
        event (e.g. a leftover chunk-size line when the response was not
        fully de-chunked by an intermediate hop), so the first event may
        NOT begin with ``event:``. dsh's SSE parser ignores that bare
        non-field line, so the stream still works — we only need to
        *recognize* the start for dedup.
        """
        for line in event.split(b"\n"):
            if line.strip() == b"event: message_start":
                return True
        return False

    def feed(self, chunk: bytes) -> list:
        """Append ``chunk``; return the list of complete events to forward
        (duplicate ``message_start``s already dropped). The incomplete tail stays
        held back until it completes or ``flush()`` is called at stream end."""
        self._buf += chunk
        out: list = []
        while True:
            idx = self._buf.find(b"\n\n")
            if idx == -1:
                break
            event = self._buf[: idx + 2]
            if not self._is_dup_start(event):
                if self._is_event_start(event):
                    self._seen = True
                    self._seen_id = self._start_id(event)
                out.append(event)
            self._buf = self._buf[idx + 2:]
        return out

    def flush(self) -> bytes:
        """Release any held-back incomplete tail as-is (stream-end leftover)."""
        tail, self._buf = self._buf, b""
        return tail


def make_handler(cfg: ShimConfig):
    """Build ``(HandlerClass, ThreadingHTTPServer)`` bound to ``cfg``."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            try:
                print(f"[shim {cfg.listen_port}] {self.command} {self.path}", flush=True)
            except Exception:
                pass

        def _send_success(self, resp, raw: bytes) -> None:
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                if k.lower() not in _STRIP_RES_HEADERS:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self._write_safe(raw)

        def _write_safe(self, data: bytes) -> None:
            """Write to the client; a client that hung up must not kill the
            handler thread (a dead thread takes the whole shim down)."""
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                self.close_connection = True

        def _send_error(self, code: int, raw: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self._write_safe(raw)

        def _send_502(self, why: str) -> None:
            msg = json.dumps(
                {"error": {"message": f"shim upstream error: {why}", "code": 502}}
            ).encode()
            self._send_error(502, msg)

        def _write_stream_chunk(self, data: bytes) -> None:
            """Forward one upstream chunk to the client immediately (flushed, so
            dsh sees tokens live instead of after the whole generation). A client
            that disconnects raises _ClientGone so the read loop can stop."""
            try:
                self.wfile.write(data)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
                self.close_connection = True
                raise _ClientGone()

        def _stream_upstream(self, resp) -> None:
            """SSE pass-through: send the upstream's 2xx headers, then pipe each
            chunk to the client as it arrives.

            A mid-stream cut can no longer be a 502 — we already sent 200 — so
            we emit a terminal SSE event and close cleanly. The client sees a
            clean end to the stream instead of a raw connection close.

            The body is forwarded with ``resp.read1(4096)``: that does ONE
            de-chunked read and returns as soon as ANY data is available
            (it does NOT wait to fill 4096). ``resp.read(4096)`` would instead
            block until the buffer is full, batching the whole generation and
            only delivering it at the end — which defeats live streaming. A
            bounded per-read flush also keeps a stalling client from wedging
            the upstream socket buffer.

            NOTE: this MUST be ``resp.read1`` (HTTPResponse), NOT
            ``resp.fp.read1`` (the raw socket). The upstream is
            ``Transfer-Encoding: chunked``; ``resp.fp`` bypasses
            de-chunking and leaks the hex chunk-size markers into the SSE
            byte stream. dsh's SSE parser only tolerates a marker that lands
            on a standalone line — when a ``data:`` JSON payload straddles a
            chunk boundary the marker lands mid-JSON and the stream fails
            with ``DeepSeek Messages SSE contains invalid JSON``.
            """
            self.send_response(resp.status)
            for k, v in resp.headers.items():
                lk = k.lower()
                # Re-frame the body: never forward the upstream's framing headers
                # (content-length / transfer-encoding) or hop-by-hop headers.
                if lk in _STRIP_RES_HEADERS or lk in ("content-length",
                                                       "transfer-encoding"):
                    continue
                self.send_header(k, v)
            self.send_header("X-Accel-Buffering", "no")  # tell any proxy: don't buffer SSE
            # The streamed body has no Content-Length and we do not add chunked
            # framing, so the ONLY way the client knows the response is complete
            # is the connection closing. Signal + enforce that, or the client
            # (dsh/undici) will sit on the keep-alive connection waiting for a
            # terminator that never comes and time out mid-stream.
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            # Route every upstream chunk through the message_start dedup filter.
            # It forwards complete events as they arrive (stream stays live) and
            # drops a duplicate message_start; the trailing incomplete event is
            # held back and released on flush() at stream end / cut.
            dedup = _SSEMessageStartDedup()

            def _emit(seq) -> None:
                for ev in seq:
                    self._write_stream_chunk(ev)

            def _push(chunk: bytes) -> None:
                if chunk:
                    _emit(dedup.feed(chunk))

            def _flush_tail() -> None:
                tail = dedup.flush()
                if tail:
                    _emit([tail])

            try:
                while True:
                    try:
                        chunk = resp.read1(4096)
                    except http.client.IncompleteRead as ir:
                        # Upstream severed the stream mid-body (declared more
                        # data than it actually sent). Forward whatever it
                        # managed to send, then emit a clean terminal event.
                        got = ir.partial or b""
                        print(f"[shim {cfg.listen_port}] upstream stream cut mid-response "
                              f"(IncompleteRead, {len(got)}b received)", flush=True)
                        if got:
                            _push(got)
                        _flush_tail()
                        self._write_stream_chunk(
                            b'data: [ERROR] upstream stream interrupted\n\n')
                        break
                    if not chunk:
                        break  # clean EOF — upstream finished the stream
                    _push(chunk)
            except _ClientGone:
                pass  # client went away — stop, nothing to log
            except (http.client.IncompleteRead, urllib.error.URLError,
                    socket.timeout, ConnectionError, OSError) as e:
                # Upstream cut the stream mid-body (ConnectionResetError for a
                # TCP RST, IncompleteRead for a truncated body) — the root
                # cause of the dsh "TRANSPORT" errors. Emit a clean terminal
                # event instead of letting the exception kill the handler.
                print(f"[shim {cfg.listen_port}] upstream stream cut mid-response: "
                      f"{type(e).__name__}: {e}", flush=True)
                try:
                    _flush_tail()
                    self._write_stream_chunk(
                        b'data: [ERROR] upstream stream interrupted\n\n')
                except _ClientGone:
                    pass
            else:
                _flush_tail()  # clean EOF: release any held-back tail

        def _do_forward_once(self, method: str, url: str, headers: dict,
                             data: Optional[bytes], stream: bool = False):
            """Send one upstream attempt.

            Returns ``None`` once a response has been written to the client (a
            success — buffered or streamed — or a terminal error). Otherwise it
            returns a tag the caller can act on:
              * ``("http", code, raw)``   — upstream answered with an HTTP error
                that has NOT yet been sent (caller may self-heal and retry).
              * ``("transport", err)``    — a pre-response transport failure
                (connect error, or a buffered read that died before any client
                byte). Nothing was sent, so the caller may retry the request.
            """
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                resp = urllib.request.urlopen(req, timeout=cfg.timeout)
            except urllib.error.HTTPError as e:
                raw = e.read()
                print(f"[shim {cfg.listen_port}] upstream HTTP {e.code}: {raw[:200]!r}",
                      flush=True)
                return ("http", e.code, raw)
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
                return ("transport", e)
            if stream:
                self._stream_upstream(resp)
                return None
            # Non-stream: buffer the whole body, but GUARD the read. The old
            # code had `raw = resp.read()` outside any try, so a mid-read cut
            # raised IncompleteRead out of the handler and killed the thread.
            try:
                raw = resp.read()
            except (http.client.IncompleteRead, urllib.error.URLError,
                    socket.timeout, ConnectionError, OSError) as e:
                print(f"[shim {cfg.listen_port}] upstream read failed pre-response: {e}",
                      flush=True)
                return ("transport", e)
            self._send_success(resp, raw)
            return None

        def _forward_models(self, method: str) -> None:
            """Handle ``GET /v1/models``: forward, but enrich the body so the
            client's context-window detection reads the upstream's REAL limit.

            The shim is the one hop every provider goes through, so this is the
            single place to bridge the field-name gap: SGLang/vLLM report the
            window as ``max_model_len`` (which dsh's detector doesn't read), so
            dsh would otherwise fall back to a hardcoded default. We copy the
            discovered window into ``context_window`` — a field dsh DOES read.

            Only the ``/models`` path is enriched (matched on the request
            path); anything else is forwarded verbatim. On a non-enrichable body
            (no window field, or not a models payload) the original bytes are
            forwarded unchanged, so this is a no-op for non-models / no-window
            upstreams.
            """
            url = cfg.upstream + self.path
            headers = _filter_headers(self.headers, _STRIP_REQ_HEADERS)
            if cfg.upstream_key:
                headers["Authorization"] = f"Bearer {cfg.upstream_key}"
            req = urllib.request.Request(url, headers=headers, method=method)
            try:
                resp = urllib.request.urlopen(req, timeout=cfg.timeout)
            except urllib.error.HTTPError as e:
                raw = e.read()
                print(f"[shim {cfg.listen_port}] upstream HTTP {e.code}: {raw[:200]!r}",
                      flush=True)
                self._send_error(e.code, raw)
                return
            except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
                self._send_502(str(e))
                return
            try:
                raw = resp.read()
            except (http.client.IncompleteRead, urllib.error.URLError,
                    socket.timeout, ConnectionError, OSError) as e:
                print(f"[shim {cfg.listen_port}] models read failed: {e}", flush=True)
                self._send_502(str(e))
                return
            enriched = enrich_models_body(raw)
            out = enriched if enriched is not None else raw
            self._send_success(resp, out)

        def _forward(self, method: str, body: Optional[bytes]) -> None:
            # GET /v1/models is special: enrich the model list so the client's
            # context-window detection reads the upstream's real limit (the
            # field-name bridge described in _forward_models). Anything else
            # falls through to the normal forward path.
            if method in ("GET", "HEAD") and self.path.rsplit("?", 1)[0].rstrip("/").endswith("/models"):
                self._forward_models(method)
                return
            url = cfg.upstream + self.path
            headers = _filter_headers(self.headers, _STRIP_REQ_HEADERS)
            if cfg.upstream_key:
                headers["Authorization"] = f"Bearer {cfg.upstream_key}"
            data: Optional[bytes] = None
            stream = False
            if method == "POST" and body is not None:
                data = body
                try:
                    stream = bool(json.loads(body).get("stream"))
                except (ValueError, UnicodeDecodeError):
                    stream = False
            elif method != "POST":
                # GET/HEAD/OPTIONS — no self-heal, one shot, never streamed.
                outcome = self._do_forward_once(method, url, headers, None, stream=False)
                if outcome is None:
                    return
                if outcome[0] == "http":
                    self._send_error(outcome[1], outcome[2])
                else:
                    self._send_502(str(outcome[1]))
                return

            orig_body = data
            # Accumulated self-heal state. A single request may need MORE THAN
            # ONE adaptation before the upstream accepts it — e.g. a strict
            # backend 400s on ``thinking`` AND the prompt overflows the context
            # window, so the retry must drop thinking *and* re-clamp max_tokens.
            # Each self-heal below is idempotent (guarded so it fires at most
            # once per request), so they can stack across up to three attempts.
            retry_force: Optional[int] = None   # pin max_tokens (context-fit)
            drop_thinking = False               # strip the thinking field
            drop_output_config = False          # strip the output_config field
            seen: set[str] = set()              # which self-heals already applied
            for attempt in (1, 2, 3):
                rewritten, changes = apply_rewrites(
                    orig_body, cfg, force_max_tokens=retry_force,
                    drop_thinking=drop_thinking,
                    drop_output_config=drop_output_config)
                for c in changes:
                    print(f"[shim {cfg.listen_port}] rewrite: {c}", flush=True)
                # log the model field (helps debug which model reached upstream)
                try:
                    _m = json.loads(rewritten).get("model")
                    if _m:
                        print(f"[shim {cfg.listen_port}] model: {_m}", flush=True)
                except Exception:
                    pass
                h = dict(headers)
                h["Content-Length"] = str(len(rewritten))
                outcome = self._do_forward_once("POST", url, h, rewritten, stream=stream)
                if outcome is None:
                    return  # a client response was written (success / stream)
                if outcome[0] == "transport":
                    # Pre-response transport failure: nothing reached the client,
                    # so a full-request resend is safe. Retry once with backoff;
                    # if that also fails, emit a clean 502.
                    if attempt == 1:
                        print(f"[shim {cfg.listen_port}] upstream transport error, "
                              f"retrying once: {outcome[1]}", flush=True)
                        time.sleep(1.0)
                        continue
                    self._send_502(str(outcome[1]))
                    return
                # outcome == ("http", code, raw)
                code, raw = outcome[1], outcome[2]
                healed = False
                # 1) CONTEXT-WINDOW: "…exceeds the model's maximum context
                #    length of W tokens. You requested a total of T tokens:
                #    I tokens from the input messages…" → the prompt plus
                #    max_tokens overflowed the TOTAL window. Retry with
                #    max_tokens = W - I - reserve (the server's EXACT input
                #    count), and persist the window so future turns clamp
                #    proactively. sglang/vLLM's max_model_len is input+output,
                #    not an output cap — so dsh's own max_tokens (window minus
                #    its input *estimate*) can still overflow the real window.
                if "context" not in seen:
                    window, in_tok, _total = _parse_context_error(raw)
                    if window is not None:
                        seen.add("context")
                        if in_tok is None:
                            in_tok = _estimate_input_tokens(
                                json.loads(rewritten))
                        retry_force = max(1, window - in_tok - 128)
                        if not cfg.context_window or cfg.context_window != window:
                            cfg.context_window = window
                        print(f"[shim {cfg.listen_port}] self-heal: context "
                              f"window={window}, input={in_tok} tokens -> retry "
                              f"max_tokens={retry_force}", flush=True)
                        _persist_provider_cap(cfg.provider_name, cfg.token_cap,
                                              window)
                        healed = True
                # 2) OUTPUT-CAP: "…at most N completion tokens" / "max_tokens
                #    is too large" → lower the output cap and retry.
                if not healed and "cap" not in seen:
                    cap = _parse_error_cap(raw)
                    if cap is not None and cap < cfg.token_cap:
                        seen.add("cap")
                        old = cfg.token_cap
                        cfg.token_cap = max(cap, 256)
                        print(f"[shim {cfg.listen_port}] self-heal: output cap "
                              f"{old}->{cfg.token_cap} (upstream reports max {cap})",
                              flush=True)
                        _persist_provider_cap(cfg.provider_name, cfg.token_cap)
                        healed = True
                # 3) EXTENDED-THINKING: upstream rejects the ``thinking`` field
                #    (budget required / no reasoning parser). The model still
                #    answers a plain message — only the reasoning trace is lost.
                #    Strip the field and retry.
                if not healed and "thinking" not in seen:
                    if _is_thinking_error(raw):
                        try:
                            _had_thinking = "thinking" in json.loads(rewritten)
                        except (ValueError, UnicodeDecodeError):
                            _had_thinking = False
                        if _had_thinking:
                            seen.add("thinking")
                            drop_thinking = True
                            print(f"[shim {cfg.listen_port}] self-heal: upstream "
                                  "rejected extended thinking -> retry without "
                                  "the thinking field", flush=True)
                            healed = True
                # 4) OPAQUE 500 + nonstandard output_config field: one backend
                #    returns an opaque "Internal server error" whenever the
                #    request carries output_config (a field dsh's pi-ai layer
                #    adds). Drop it and retry. Only fires on a 500 AND only if
                #    the field is actually in the request, so a genuine 500
                #    without the field is NOT masked.
                if not healed and "output_config" not in seen:
                    if code == 500:
                        try:
                            _had_oc = "output_config" in json.loads(rewritten)
                        except (ValueError, UnicodeDecodeError):
                            _had_oc = False
                        if _had_oc:
                            seen.add("output_config")
                            drop_output_config = True
                            print(f"[shim {cfg.listen_port}] self-heal: upstream "
                                  "500'd with output_config present -> retry "
                                  "without the output_config field", flush=True)
                            healed = True
                if healed:
                    continue
                self._send_error(code, raw)
                return

        def do_GET(self) -> None:
            self._forward("GET", None)

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(n) if n else b""
            self._forward("POST", body)

        def do_HEAD(self) -> None:
            self._forward("GET", None)

        def do_OPTIONS(self) -> None:
            self._forward("GET", None)

    return Handler, ThreadingHTTPServer


def serve(cfg: ShimConfig) -> None:
    """Run the shim as a threaded HTTP server (blocks until interrupted)."""
    Handler, ThreadingHTTPServer = make_handler(cfg)
    httpd = ThreadingHTTPServer((cfg.listen_host, cfg.listen_port), Handler)
    httpd.daemon_threads = True
    print(
        f"[shim] listening on http://{cfg.listen_host}:{cfg.listen_port} "
        f"-> {cfg.upstream} (effort={cfg.effort_mode}, token_cap={cfg.token_cap})",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
