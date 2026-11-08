"""ReaxKit external runner for INDEEDopt's bounded JAX sampling workflow."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil


def read_request(path):
    request = json.loads(Path(path).read_text())
    if request.get('contract') != 'reaxkit-external-runner-v1' or request.get('optimizer') != 'indeedopt':
        raise ValueError('Expected an INDEEDopt reaxkit-external-runner-v1 request')
    settings = request['settings']
    if settings.get('engine') != 'jax-md' or not isinstance(settings.get('short_samples'), int) or settings['short_samples'] < 1:
        raise ValueError('This runner supports only engine=jax-md and short_samples >= 1; not full deep-learning optimization')
    allowed = {'engine', 'short_samples', 'jax_python', 'jaxreaxff_source', 'jaxmd_source'}
    unsupported = set(request.get('explicit_settings', [])) - allowed
    if unsupported:
        raise ValueError(f'Unsupported explicit settings for short mode: {sorted(unsupported)}')
    if request['inputs'].get('control'):
        raise ValueError('JAX short mode does not read control; supply control: null')
    for key in ('jax_python', 'jaxreaxff_source', 'jaxmd_source'):
        if not settings.get(key) or not Path(settings[key]).exists():
            raise ValueError(f'Missing {key}')
    if request.get('params_bounds_order') not in ('min_max', 'max_min'):
        raise ValueError('Explicit params_bounds_order is required')
    return request


def input_parameters(request):
    rows = []
    seen = set()
    for line in Path(request['inputs']['params']).read_text().splitlines():
        fields = re.split(r'[!#]', line, maxsplit=1)[0].split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError('params requires six columns')
        pointer = tuple(int(value) for value in fields[:3])
        values = [float(value.replace('D', 'E').replace('d', 'e')) for value in fields[3:]]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Nonfinite parameter values')
        step, lower, upper = values
        if request['params_bounds_order'] == 'max_min':
            lower, upper = upper, lower
        if lower >= upper or pointer in seen:
            raise ValueError('Invalid bounds or duplicate parameter pointer')
        seen.add(pointer)
        rows.append([*pointer, step, lower, upper])
    if not rows:
        raise ValueError('No parameters supplied')
    return rows


def render_field(text, rows, values):
    lines = text.splitlines(keepends=True)
    atom_header = next(index for index, line in enumerate(lines) if 'Nr of atoms' in line)
    atom_count = int(lines[atom_header].split()[0])
    atom_start = atom_header + 4
    bond_header = atom_start + 4 * atom_count
    if 'Nr of bonds' not in lines[bond_header]:
        raise ValueError('Only classic four-line atom/two-line bond layouts are supported')
    bond_count = int(lines[bond_header].split()[0])
    for row, value in zip(rows, values, strict=True):
        section, entry, parameter = row[:3]
        if section == 2 and 1 <= entry <= atom_count and 1 <= parameter <= 32:
            offset, column = divmod(parameter - 1, 8)
            line_index = atom_start + 4 * (entry - 1) + offset
            column += int(offset == 0)
        elif section == 3 and 1 <= entry <= bond_count and 1 <= parameter <= 16:
            offset, column = divmod(parameter - 1, 8)
            line_index = bond_header + 2 + 2 * (entry - 1) + offset
            column += 2 * int(offset == 0)
        else:
            raise ValueError('This runner supports only classic atomic/bond parameter pointers (sections 2/3)')
        rounded = float(f'{value:.4f}')
        if not math.isfinite(rounded) or not row[4] <= rounded <= row[5]:
            raise ValueError('Rounded candidate is outside supplied bounds')
        tokens = list(re.finditer(r'\S+', lines[line_index]))
        token = tokens[column]
        float(token.group())
        lines[line_index] = lines[line_index][:token.start()] + f'{rounded:.4f}' + lines[line_index][token.end():]
    return ''.join(lines)


def run(request, sampler=None, evaluator=None):
    rows = input_parameters(request)
    field = Path(request['inputs']['ffield']).read_text()
    render_field(field, rows, [(row[4] + row[5]) / 2 for row in rows])
    root = Path(request['native_directory']).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'short_results').exists() or any(root.glob('ffield-*')):
        raise ValueError('Use a fresh campaign; existing candidate/results directories are not overwritten')
    if sampler is None:
        from ffield_generator.parameter_generator import parameter_generator
        sampler = parameter_generator
    if evaluator is None:
        from ffield_generator.jax_md_engine import run_engine
        evaluator = run_engine
    samples = sampler(rows, request['settings']['short_samples'])
    environment_keys = {'jax_python': 'INDEEDOPT_JAX_PYTHON', 'jaxreaxff_source': 'INDEEDOPT_JAXREAXFF_SOURCE',
                        'jaxmd_source': 'INDEEDOPT_JAXMD_SOURCE'}
    previous = {name: os.environ.get(name) for name in environment_keys.values()}
    output = root / 'short_results'
    output.mkdir()
    records = []
    best = math.inf
    try:
        for key, name in environment_keys.items():
            os.environ[name] = request['settings'][key]
        for sample in range(request['settings']['short_samples']):
            candidate = root / f'ffield-{sample}'
            candidate.mkdir()
            (candidate / 'ffield').write_text(render_field(field, rows, samples[sample]))
            for key, filename in [('geo', 'geo'), ('trainset', 'trainset.in')]:
                shutil.copyfile(request['inputs'][key], candidate / filename)
            (candidate / 'params-ml').write_text(''.join(' '.join(map(str, row)) + '\n' for row in rows))
            (candidate / 'parameters').write_text(''.join(f'{value:.4f}\n' for value in samples[sample]))
            evaluator(candidate)
            report = json.loads((candidate / 'jax_md_results/evaluation.json').read_text())
            loss = float(report['loss'])
            if not math.isfinite(loss) or loss < 0 or not report.get('items'):
                raise ValueError('Invalid engine evaluation report')
            records.append((sample, loss))
            temporary = output / 'results.csv.tmp'
            with temporary.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['sample', 'loss'])
                writer.writerows(records)
            temporary.replace(output / 'results.csv')
            if loss < best:
                for source, target in [('ffield', 'best_ffield'), ('parameters', 'best_parameters'),
                                       ('jax_md_results/evaluation.json', 'best_report.json')]:
                    temporary = output / (target + '.tmp')
                    shutil.copyfile(candidate / source, temporary)
                    temporary.replace(output / target)
                best = loss
            print(f'Sample {sample}: {loss}', flush=True)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_config', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        request = read_request(args.run_config)
        if args.check_only:
            rows = input_parameters(request)
            render_field(Path(request['inputs']['ffield']).read_text(), rows,
                         [(row[4] + row[5]) / 2 for row in rows])
            print('Runner input checks passed; engine execution remains unverified')
        else:
            run(request)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'INDEEDopt runner: {error}\n')


if __name__ == '__main__':
    main()
