"""Command line entry point: fetch data, run the pipeline, score a flight, launch the app."""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tailslip", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download and verify the source tables").add_argument("--raw", default="data/raw")
    r = sub.add_parser("run", help="build features, fit, evaluate, write outputs and charts")
    r.add_argument("--raw", default="data/raw")
    r.add_argument("--out", default="outputs")
    r.add_argument("--assets", default="assets")
    r.add_argument("--quick", action="store_true", help="smaller studies, for smoke tests")
    sub.add_parser("app", help="launch the Gradio replay app")
    s = sub.add_parser("show", help="print the headline metrics from outputs/metrics.json")
    s.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        from tailslip import data

        print(data.fetch(args.raw))
    elif args.cmd == "run":
        from tailslip import data, pipeline

        data.fetch(args.raw)
        pipeline.run(args.raw, args.out, args.assets, quick=args.quick)
    elif args.cmd == "app":
        from tailslip import app

        app.main()
    elif args.cmd == "show":
        m = json.load(open(f"{args.out}/metrics.json"))
        print(
            json.dumps(
                {"shipped": m["selection"]["shipped"], "test": m["shipped_test"], "economics": m["economics"]["test"]}, indent=2
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
