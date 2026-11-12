"""Gradio front end that replays held-out flights through the saved model."""

from __future__ import annotations

import pandas as pd

from tailslip import scoring


def build_app(bundle_path="outputs/tailslip_model.joblib", replay_path="outputs/test_replay.parquet"):
    import gradio as gr

    bundle = scoring.load_bundle(bundle_path)
    replay = pd.read_parquet(replay_path)
    feat_cols = [c for c in replay.columns if c.startswith("f_")]
    dates = sorted(replay["sched_date"].dt.strftime("%Y-%m-%d").unique())

    def choices(origin, date):
        day = replay[(replay["origin"] == origin) & (replay["sched_date"].dt.strftime("%Y-%m-%d") == date)]
        day = day.sort_values("sched_utc")
        return gr.update(
            choices=[
                f"{r.carrier}{r.flight} to {r.dest} at {int(r.hour):02d}:{int(r.minute):02d}" + f" [{r.flight_id}]"
                for r in day.itertuples()
            ],
            value=None,
        )

    def run(label, prev_delay, recent_delay):
        if not label:
            return "Pick a flight first.", ""
        fid = int(label.rsplit("[", 1)[1].rstrip("]"))
        rec = replay[replay["flight_id"] == fid].iloc[0]
        row = pd.DataFrame([rec[feat_cols].to_numpy()], columns=[c[2:] for c in feat_cols])
        for c in row.columns:
            if c not in ("carrier", "origin", "dest", "manufacturer", "plane_type"):
                row[c] = pd.to_numeric(row[c])
        row = scoring.apply_overrides(
            row,
            {
                "prev_dep_delay": prev_delay if prev_delay is not None and prev_delay != -999 else None,
                "recent_mean_delay": recent_delay if recent_delay is not None and recent_delay != -999 else None,
            },
        )
        out = scoring.summarise(bundle, row, int(rec["hour"]), int(rec["minute"]))
        verdict = "send a delay alert" if out["alert"] else "no alert"
        head = (
            f"Chance of leaving 15+ minutes late: {out['p_delay']:.0%} ({verdict}).\n"
            f"Four times in five it leaves by {out['p80_departs_by']}; median {out['p50_departs_by']}."
        )
        why = "\n".join(f"- {r['text']}" for r in out["reasons"])
        actual = rec["dep_delay"]
        actual_txt = "cancelled" if pd.isna(actual) else f"{actual:.0f} min vs schedule"
        return head, why + f"\n\nWhat happened: {actual_txt}."

    with gr.Blocks(title="tailslip") as demo:
        gr.Markdown("### tailslip\nReplay a held-out flight as it looked two hours before departure.")
        with gr.Row():
            origin = gr.Dropdown(["EWR", "JFK", "LGA"], value="JFK", label="Airport")
            date = gr.Dropdown(dates, value=dates[0], label="Date")
        flight = gr.Dropdown([], label="Flight")
        with gr.Row():
            prev = gr.Slider(-999, 240, value=-999, step=1, label="What-if: previous leg delay, min (-999 keeps actual)")
            recent = gr.Slider(-999, 120, value=-999, step=1, label="What-if: recent airport delay, min (-999 keeps actual)")
        go = gr.Button("Score")
        head = gr.Textbox(label="Forecast", lines=3)
        why = gr.Textbox(label="Why", lines=6)
        origin.change(choices, [origin, date], flight)
        date.change(choices, [origin, date], flight)
        go.click(run, [flight, prev, recent], [head, why])
    return demo


def main():  # pragma: no cover
    build_app().launch()


if __name__ == "__main__":  # pragma: no cover
    main()
