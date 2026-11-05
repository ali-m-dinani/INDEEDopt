import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _format_field(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("JAX-ReaxFF returned a non-finite value")
    field = f"{value:11.4E}"
    if len(field) != 11:
        raise ValueError(f"Value does not fit the fort.99 field: {value}")
    return field


def write_legacy_results(report, directory):
    directory = Path(directory)
    items = report["items"]
    loss = float(report["loss"])
    if not items or not math.isfinite(loss):
        raise ValueError("JAX-ReaxFF returned an empty or invalid evaluation")

    lines = ["JAX-ReaxFF evaluation\n", "Prediction, target, and weight\n"]
    for item in items:
        prediction = _format_field(item["prediction"])
        target = _format_field(item["target"])
        weight = _format_field(item["weight"])
        if float(item["weight"]) == 0:
            raise ValueError("JAX-ReaxFF returned a zero training weight")
        lines.append((" " * 61 + prediction + " " + target + " " + weight).ljust(120) + "\n")

    (directory / "fort.99").write_text("".join(lines), encoding="ascii", newline="\n")
    (directory / "fort.13").write_text(_format_field(loss) + "\n", encoding="ascii", newline="\n")


def run_engine(directory="."):
    directory = Path(directory)
    output_dir = directory / "jax_md_results"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "evaluation.json"
    command = [
        os.environ.get("INDEEDOPT_JAX_PYTHON", sys.executable), "-m", "jaxreaxff.driver",
        "--init_FF", "ffield", "--params", "params-ml",
        "--geo", "geo", "--train_file", "trainset.in",
        "--num_trials", "0", "--out_folder", "jax_md_results",
        "--machine_report", str(Path("jax_md_results") / "evaluation.json"),
    ]
    environment = os.environ.copy()
    source = environment.get("INDEEDOPT_JAXREAXFF_SOURCE")
    jax_md_source = environment.get("INDEEDOPT_JAXMD_SOURCE")
    if source or jax_md_source:
        environment["PYTHONPATH"] = os.pathsep.join(filter(
            None, [jax_md_source, source, environment.get("PYTHONPATH")]
        ))
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=directory, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    write_legacy_results(report, directory)
