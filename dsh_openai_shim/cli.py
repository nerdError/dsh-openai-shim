"""Command-line entrypoint for dsh-openai-shim.

Examples
--------
Run the shim with defaults (map high->medium, token cap 100000):

    dsh-openai-shim serve --upstream https://host/path

Custom effort handling + explicit upstream key:

    dsh-openai-shim serve --upstream https://host/path \
        --effort-mode drop --token-cap 128000 --port 8090

One-shot, no server — apply the rewrites to a JSON file / stdin (for tests):

    cat req.json | dsh-openai-shim rewrite --effort-mode map
"""
from __future__ import annotations

import argparse
import os
import sys

from . import ShimConfig, apply_rewrites, serve


def _cfg_from_args(args) -> ShimConfig:
    return ShimConfig(
        upstream=args.upstream,
        listen_host=args.host,
        listen_port=args.port,
        effort_mode=args.effort_mode,
        token_cap=args.token_cap,
        upstream_key=args.upstream_key,
    )


def _env(name: str, default=None):
    return os.environ.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dsh-openai-shim",
        description="OpenAI-compatible proxy that adapts DeepSeek Harness to any endpoint.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the proxy")
    s.add_argument("--upstream", default=_env("DSH_SHIM_UPSTREAM"),
                   help="upstream base URL (no trailing /v1 if the client appends it). "
                        "Falls back to $DSH_SHIM_UPSTREAM.")
    s.add_argument("--host", default=_env("DSH_SHIM_HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=_env("DSH_SHIM_PORT", 8090))
    s.add_argument("--effort-mode", choices=["map", "drop", "off", "low", "none"],
                   default=_env("DSH_SHIM_EFFORT_MODE", "map"))
    s.add_argument("--token-cap", type=int, default=_env("DSH_SHIM_TOKEN_CAP", 100000),
                   help="clamp completion max_tokens to this (0 = off)")
    s.add_argument("--upstream-key", default=_env("DSH_SHIM_UPSTREAM_KEY"),
                   help="force an Authorization Bearer key upstream ($DSH_SHIM_UPSTREAM_KEY)")
    s.set_defaults(func=_cmd_serve)

    r = sub.add_parser("rewrite", help="apply rewrites to a JSON request body (stdin -> stdout)")
    r.add_argument("--effort-mode", choices=["map", "drop", "off", "low", "none"], default="map")
    r.add_argument("--token-cap", type=int, default=100000)
    r.set_defaults(func=_cmd_rewrite)

    return p


def _cmd_serve(args) -> int:
    if not args.upstream:
        print("dsh-openai-shim: --upstream (or $DSH_SHIM_UPSTREAM) is required to serve.",
              file=sys.stderr)
        return 2
    serve(_cfg_from_args(args))
    return 0


def _cmd_rewrite(args) -> int:
    cfg = ShimConfig(upstream="http://upstream", effort_mode=args.effort_mode, token_cap=args.token_cap)
    body = sys.stdin.buffer.read()
    new, changes = apply_rewrites(body, cfg)
    for c in changes:
        print(f"rewrite: {c}", file=sys.stderr)
    sys.stdout.buffer.write(new)
    sys.stdout.buffer.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
