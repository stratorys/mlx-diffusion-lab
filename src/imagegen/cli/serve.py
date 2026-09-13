"""Run the HTTP server.

    imagegen-serve --profile plain-4b --runner mflux --port 4242

One profile per process. Serving both means two processes on two ports, which is
deliberate: adapters are baked into the model at construction, so switching profiles
means reloading everything.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from imagegen.api.app import create_app

DEFAULT_PORT = 4242


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default=None, help="defaults to IMAGEGEN_PROFILE, else the first")
    parser.add_argument("--runner", default=None, help="overrides IMAGEGEN_RUNNER")
    parser.add_argument("--catalogue", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv), parser


def main(argv: list[str] | None = None) -> int:
    args, _ = parse_args(argv)
    app = create_app(args.profile, args.runner, args.catalogue, args.out)
    service = app.state.service
    print(f"profile {service.profile.name}, runner {service.runner}", flush=True)
    print(f"output {service.out_dir}", flush=True)
    print(f"http://{args.host}:{args.port}/docs", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
