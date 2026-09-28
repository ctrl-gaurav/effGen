"""Argument declarations for the server, monitoring and diagnostic commands.

``effgen.cli._main.create_parser`` calls these in the order the top-level
``--help`` lists them; each attaches one command (or one alias pair) to the
given subparsers action and returns nothing.
"""

from __future__ import annotations

import argparse


def add_serve_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen serve`` — the API server and its bind/rate-limit flags."""
    serve_parser = subparsers.add_parser(
        'serve', help='Start API server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Operational settings (environment variables):\n"
            "  EFFGEN_API_KEY        static API key (Bearer or X-API-Key). If unset\n"
            "                        and not in dev mode, an ephemeral key is minted\n"
            "                        and printed once — never unauthenticated.\n"
            "  EFFGEN_DEV_MODE=1     disable auth (loud warning; local dev only).\n"
            "  EFFGEN_RATE_LIMIT     requests/minute per client IP (0 disables;\n"
            "                        health probes are always exempt). Or use\n"
            "                        --rate-limit.\n"
            "  EFFGEN_TRUST_PROXY=1  trust the first X-Forwarded-For hop as the\n"
            "                        rate-limit client IP (default: off — the raw\n"
            "                        socket peer is used, since a caller can set\n"
            "                        that header to anything). Enable only behind\n"
            "                        a reverse proxy that sets/overwrites it.\n"
            "  EFFGEN_CORS_ORIGINS   comma-separated allowed origins (default: none;\n"
            "                        cross-origin is fail-closed for a backend API).\n"
            "  EFFGEN_OIDC_ISSUER /  enable OIDC/JWT auth instead of a static key.\n"
            "  EFFGEN_OIDC_CLIENT_ID\n"
            "  EFFGEN_PUBLIC_METRICS=1   serve /metrics without auth (default: auth).\n"
            "  EFFGEN_PUBLIC_DASHBOARD=1 serve /dashboard/data.json + /dashboard/spans\n"
            "                        without auth, for local viewing (default: auth;\n"
            "                        the /dashboard page itself always loads, but its\n"
            "                        data calls 401 without this or an API key).\n"
            "  EFFGEN_MODEL_POOL_SIZE    loaded models kept warm (default 4).\n"
            "  EFFGEN_NO_DOTENV=1    skip the .env filesystem search entirely, so\n"
            "                        only environment variables the orchestrator set\n"
            "                        are visible (EFFGEN_DOTENV=none is equivalent).\n"
            "\n"
            "Watching a running server: `effgen top` reads this server's\n"
            "/dashboard/data.json. Point it with --url/--port, or export\n"
            "EFFGEN_SERVER_URL so it finds this server with no flags.\n"
            "\n"
            "Scaling: `effgen serve` runs a single worker. For multiple workers,\n"
            "run the app factory under uvicorn/gunicorn, e.g.:\n"
            "  uvicorn effgen.server.app:create_app --factory --workers 4 --port 8000\n"
        ),
    )
    serve_parser.add_argument(
        '--host', default='127.0.0.1',
        help='Host to bind to (default 127.0.0.1, loopback-only). '
             'Pass --host 0.0.0.0 to expose on all interfaces (set EFFGEN_API_KEY first).',
    )
    serve_parser.add_argument('-p', '--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument(
        '--rate-limit', type=int, default=None, metavar='N',
        help='Requests/minute per client IP (overrides EFFGEN_RATE_LIMIT; '
             '0 disables). Health probes are always exempt.',
    )
    serve_parser.add_argument(
        '--trust-proxy', action='store_true', default=None,
        help='Trust the first X-Forwarded-For hop as the rate-limit client IP '
             '(overrides EFFGEN_TRUST_PROXY). Enable only behind a reverse '
             'proxy that sets/overwrites this header — otherwise any caller '
             'can spoof it to bypass the rate limit.',
    )

def add_top_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen top`` and its ``monitor`` alias from one block of flags."""
    from effgen.cli.monitor import (
        DEFAULT_ACTIVITY_LIMIT as _MON_LIMIT,
    )
    from effgen.cli.monitor import (
        DEFAULT_INTERVAL as _MON_INTERVAL,
    )
    _top_epilog = (
        "Examples:\n"
        "  effgen top                          # live view, refreshing in place\n"
        "  effgen top --interval 1             # faster refresh\n"
        "  effgen top --url http://host:8000   # watch a remote server\n"
        "  effgen top --once                   # one static snapshot\n"
        "  effgen top --json | jq .spend       # machine-readable snapshot\n"
        "\n"
        "Panels and their sources:\n"
        "  Activity    completed runs from the local run history (all processes)\n"
        "  Traffic     the server's own counters, since that process started\n"
        "  Per-model   per-model calls, errors, p95 latency and cost from the server\n"
        "  Spend       the local cost ledger: 24h total, daily budget, $/h burn\n"
        "  GPU         physical device memory and utilization, across all processes\n"
        "\n"
        "Each panel names the window and process it measures; figures from\n"
        "different sources are never combined. Activity, Spend and GPU need no\n"
        "server, so the view is useful on a host running only local agents; the\n"
        "server-backed panels then read as unavailable, naming the URL tried.\n"
        "\n"
        "The server URL comes from --url, --port, EFFGEN_SERVER_URL, or\n"
        "http://127.0.0.1:8000. Reading a server's data needs its API key unless\n"
        "it was started with EFFGEN_PUBLIC_DASHBOARD=1; pass --api-key or set\n"
        "EFFGEN_API_KEY.\n"
        "\n"
        "Piped output, --json, --once, --no-animation, NO_COLOR and\n"
        "EFFGEN_NO_ANIM=1 all print a single snapshot and exit instead of taking\n"
        "over the screen. On a terminal, q or Ctrl-C quits.\n"
    )
    for _top_name, _top_help in (
        ('top', 'Live terminal view of runs, traffic, spend and GPU'),
        ('monitor', 'Alias for `effgen top`'),
    ):
        _top_parser = subparsers.add_parser(
            _top_name, help=_top_help,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=_top_epilog,
        )
        _top_parser.add_argument('--json', dest='output_json', action='store_true',
                                 help='Print one snapshot as JSON and exit')
        _top_parser.add_argument('--once', action='store_true',
                                 help='Print one static snapshot and exit (no refresh loop)')
        _top_parser.add_argument('--interval', type=float, default=_MON_INTERVAL, metavar='SECONDS',
                                 help=f'Seconds between refreshes (default: {_MON_INTERVAL:g})')
        _top_parser.add_argument('--count', type=int, default=None, metavar='N',
                                 help='Stop after N refreshes (default: run until you quit)')
        _top_parser.add_argument('--limit', type=int, default=_MON_LIMIT, metavar='N',
                                 help=f'Runs to show in the activity panel (default: {_MON_LIMIT})')
        _top_parser.add_argument('--url', metavar='URL',
                                 help='Server base URL (default: EFFGEN_SERVER_URL '
                                      'or http://127.0.0.1:8000)')
        _top_parser.add_argument('-p', '--port', type=int, metavar='PORT',
                                 help='Server port on 127.0.0.1 (shorthand for --url)')
        _top_parser.add_argument('--api-key', dest='api_key', metavar='KEY',
                                 help='API key for the server (default: EFFGEN_API_KEY)')
        _top_parser.add_argument('--no-animation', action='store_true', default=argparse.SUPPRESS,
                                 help='Print one static snapshot instead of the live view')

def add_health_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen health`` — infrastructure checks, network opt-in."""
    health_parser = subparsers.add_parser(
        'health', help='Check effGen infrastructure health (contacts external services)')
    health_parser.add_argument(
        '--remote', '--online', dest='health_remote', action='store_true',
        help='Opt in to network checks (effgen.org / PyPI). Without this, health '
             'does not contact any external service. EFFGEN_HEALTH_REMOTE=1 also enables it.')

def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen doctor`` — provider keys, system and coding readiness."""
    doctor_parser = subparsers.add_parser(
        'doctor',
        help='Check provider keys, the system, and what effgen code needs',
        description='Check which providers are keyed, report the CUDA/torch/vLLM '
                    'state, and check what `effgen code` needs from this machine: '
                    'a writable workspace, a sandbox backend for the code it runs, '
                    'and git for repository context.',
    )
    doctor_parser.add_argument('--json', dest='output_json', action='store_true',
                               help='Output as JSON')
    doctor_parser.add_argument('-w', '--workspace', metavar='DIR',
                               help='Check the coding workspace for DIR instead of '
                                    'EFFGEN_WORKSPACE / the current directory')
    doctor_parser.add_argument('--provider', dest='doctor_provider',
                               help='Check a specific provider only')
    doctor_parser.add_argument('--live', action='store_true',
                               help='Make a tiny live call per keyed provider to confirm the '
                                    'default model is actually usable (not just that a key exists)')
    doctor_parser.add_argument('--cheap', action='store_true',
                               help='With --live, use the cheapest/default model and minimal tokens')
    doctor_parser.add_argument('--probe', dest='probe_model', metavar='MODEL',
                               help='Measure what MODEL does when handed a tool (a model '
                                    'served at --base-url, or a local model) and print it; '
                                    'the result is stored and reused by agents at '
                                    'tool_calling_mode="auto"')
    doctor_parser.add_argument('--base-url', dest='probe_base_url', metavar='URL',
                               help='With --probe, the OpenAI-compatible endpoint serving MODEL')
    doctor_parser.add_argument('--api-key-env', dest='probe_api_key_env', metavar='VAR',
                               help='With --probe, the environment variable holding the '
                                    "endpoint's key")
    doctor_parser.add_argument('--refresh', dest='probe_refresh', action='store_true',
                               help='With --probe, measure again and replace the stored result')

def add_plugin_parser(subparsers: argparse._SubParsersAction) -> None:
    """Declare ``effgen create-plugin`` — generate a plugin project scaffold."""
    plugin_parser = subparsers.add_parser('create-plugin', help='Generate a plugin project scaffold')
    plugin_parser.add_argument('plugin_name', help='Plugin name (e.g. my_tools)')
    plugin_parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
