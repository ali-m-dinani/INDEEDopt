# INDEEDopt: INitial DEsign Enhanced Deep learning OPTimization framework for high dimensional ReaxFF force field parameterization

![INDEEDopt Workflow Scheme](images/INDEEDopt_diagram2.png)


[![Python](https://img.shields.io/badge/Python-3.5%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.1%2B-orange)](https://pytorch.org/)
[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# UPDATES

#### Due to time limitations in other projects, I won't be able to maintain this package. ####

A common error that users of INDEEDopt face is the generation of lousy ffield files. The reason for that is the extra spaces in ffield file that are located in the "folder". Please clean all extra spaces after each line in ffield file.

Added an example INDEEDopt folder named example_github.zip. (November 2021)

Recent updates to some subroutines due to Tensorflow Deep Learning library updates (Feb 2021)

## JAX MD ReaxFF evaluation

### ReaxKit bounded smoke runner

`python reaxkit_runner.py /absolute/path/run-config.json` implements the
`reaxkit-external-runner-v1` contract for **JAX short sampling only**. Use
`runtime.mode: external_runner` and command
`[/path/to/python, /path/to/INDEEDopt/reaxkit_runner.py, "{run_config}"]`.
Install `requirements-smoke.txt` in a separate Python environment; provide the
existing JAX-ReaxFF interpreter as `settings.jax_python`, plus
`jaxreaxff_source` and `jaxmd_source`. Set `engine: jax-md`, `short_samples: 2`,
and `inputs.control: null`. No ReaxKit import or TensorFlow installation is needed.

The runner reuses INDEEDopt's Latin-hypercube sampler and existing JAX evaluator.
It honors supplied bounds (unlike the legacy main script's automatic 5% rewrite),
and supports classic four-line atomic and two-line bond parameter blocks only
(sections 2/3). Its token-based writer avoids the original byte-offset writer's
dependence on trailing spaces. Candidate values retain four-decimal precision.
Unsupported explicit full-mode settings are rejected rather than ignored.
The existing JAX evaluator defaults, including no geometry minimization, remain
unchanged; do not assume equivalence with AMS or a control-driven calculation.

Use `--check-only` with the same request to inspect runner input compatibility
without importing the sampler or launching an engine. ReaxKit YAML validation
alone does not execute this external preflight. Every run needs a fresh native
directory. Completed samples and best artifacts are atomically published in
`short_results/`; individual engine reports remain in `ffield-*/jax_md_results`.
The ReaxKit INDEEDopt converter already reads this layout. Requested orchestrator
seeds are not applied by this runner; no reproducibility claim is made.

This is **not** the deep-learning/surrogate/brute-force workflow. Those stages
and the native standalone-engine runner remain unsupported by this entrypoint.
Run real evaluations only in an authorized compute allocation.

INDEEDopt can use the `reaxff_dev` JAX MD engine through the local JAX-ReaxFF driver. This mode evaluates each INDEEDopt candidate with JAX-ReaxFF and converts its unrounded results to the legacy `fort.13` and `fort.99` files used by the optimizer. It requires the companion `--machine_report` change in `JAX-ReaxFF/jaxreaxff/driver.py`.

Install JAX-ReaxFF and its `cagrikymk/jax-md` `reaxff_dev` dependency in a Python environment, then set these variables before launching INDEEDopt from a directory containing `folder/` and `params-ml`:

```powershell
$env:INDEEDOPT_ENGINE = 'jax-md'
$env:INDEEDOPT_JAX_PYTHON = 'C:\Users\alimo\PycharmProjects\JAX-ReaxFF\.venv\Scripts\python.exe'
$env:INDEEDOPT_JAXREAXFF_SOURCE = 'C:\Users\alimo\PycharmProjects\JAX-ReaxFF'
$env:INDEEDOPT_JAXMD_SOURCE = 'C:\Users\alimo\PycharmProjects\jax-md'
python -m scoop ..\src\main-parallel.py
```

The `folder/` directory needs `ffield`, `geo`, `trainset.in`, and `params-ml`; the native ReaxFF `exe` is not used in this mode. Candidate evaluation logs and reports are written under each candidate's `jax_md_results/` directory until INDEEDopt cleans up candidate folders. INDEEDopt retains its usual error and parameter results. JAX-ReaxFF supports evaluation without optimization through `--num_trials 0`; INDEEDopt still chooses the candidate parameters.

### Short Windows trial

From the INDEEDopt repository root in PowerShell, make a fresh copy of the example inputs and evaluate two parameter samples:

```powershell
$repo = (Get-Location).Path
$run = Join-Path $repo 'short_run'
New-Item -ItemType Directory -Path $run -Force | Out-Null
Copy-Item .\example_github\folder $run -Recurse
Copy-Item .\example_github\params-ml $run
$env:INDEEDOPT_ENGINE = 'jax-md'
$env:INDEEDOPT_SHORT_SAMPLES = '2'
$env:INDEEDOPT_JAX_PYTHON = (Resolve-Path '..\JAX-ReaxFF\.venv\Scripts\python.exe').Path
$env:INDEEDOPT_JAXREAXFF_SOURCE = (Resolve-Path '..\JAX-ReaxFF').Path
$env:INDEEDOPT_JAXMD_SOURCE = (Resolve-Path '..\jax-md').Path
Set-Location $run
& (Join-Path $repo '.venv\Scripts\python.exe') (Join-Path $repo 'src\main-parallel.py')
```

Short mode runs samples sequentially and exits. `short_run/short_results/results.csv` lists every loss; `best_ffield`, `best_parameters`, and `best_report.json` contain the best sample. The `ffield-*` candidate directories retain their detailed JAX-ReaxFF reports. Use a new run directory for another trial. This is a bounded parameter search; it does not run INDEEDopt's deep-learning or brute-force stages.

