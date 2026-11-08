import json
from pathlib import Path
import tempfile
import unittest

import reaxkit_runner as runner


def field_text():
    return ('Synthetic\n0 ! general\n1 ! Nr of atoms\nheader\nheader\nheader\n'
            'Co ' + ' '.join(['1.0000'] * 8) + '\n' +
            (' '.join(['1.0000'] * 8) + '\n') * 3 +
            '1 ! Nr of bonds\nheader\n1 1 ' + ' '.join(['1.0000'] * 8) + '\n' +
            ' '.join(['1.0000'] * 8) + '\n0 ! other\n')


class RunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name, text in [('ffield', field_text()), ('geo', 'synthetic'),
                           ('trainset', 'synthetic'), ('params', '2 1 1 .1 0.5 2\n3 1 16 .1 0.5 2\n')]:
            (self.root / name).write_text(text)
        self.request = {'contract': 'reaxkit-external-runner-v1', 'optimizer': 'indeedopt',
                        'inputs': {name: str(self.root / name) for name in ('ffield', 'geo', 'trainset', 'params')},
                        'params_bounds_order': 'min_max', 'explicit_settings': ['engine', 'short_samples'],
                        'settings': {'engine': 'jax-md', 'short_samples': 2,
                                     'jax_python': str(self.root / 'ffield'),
                                     'jaxreaxff_source': str(self.root), 'jaxmd_source': str(self.root)},
                        'native_directory': str(self.root / 'native')}

    def test_request_rejects_unsupported_modes(self):
        path = self.root / 'request.json'
        path.write_text(json.dumps(self.request))
        self.assertEqual(runner.read_request(path)['optimizer'], 'indeedopt')
        for key, value in [('engine', 'reaxff'), ('short_samples', 0)]:
            request = json.loads(json.dumps(self.request))
            request['settings'][key] = value
            path.write_text(json.dumps(request))
            with self.assertRaises(ValueError):
                runner.read_request(path)
        self.request['explicit_settings'].append('lm_iter')
        path.write_text(json.dumps(self.request))
        with self.assertRaisesRegex(ValueError, 'Unsupported explicit'):
            runner.read_request(path)

    def test_render_and_bounds(self):
        rows = runner.input_parameters(self.request)
        rendered = runner.render_field(field_text(), rows, [1.5, 1.7])
        self.assertIn('Co 1.5000 1.0000', rendered)
        self.assertIn('1.7000\n0 ! other', rendered)
        with self.assertRaisesRegex(ValueError, 'outside'):
            runner.render_field(field_text(), rows, [3, 1])
        self.request['params_bounds_order'] = 'max_min'
        with self.assertRaisesRegex(ValueError, 'bounds'):
            runner.input_parameters(self.request)
        (self.root / 'params').write_text('2 1 1 .1 2 0.5\n')
        self.assertEqual(runner.input_parameters(self.request)[0][-2:], [0.5, 2])

    def test_checkpoint_and_no_overwrite(self):
        def evaluate(candidate):
            output = candidate / 'jax_md_results'
            output.mkdir()
            loss = 2 if candidate.name == 'ffield-0' else 1
            (output / 'evaluation.json').write_text(json.dumps({'loss': loss, 'items': [{}]}))
        runner.run(self.request, sampler=lambda rows, count: {0: [1, 1], 1: [1.5, 1.7]}, evaluator=evaluate)
        output = self.root / 'native/short_results'
        self.assertIn('1,1.0', (output / 'results.csv').read_text())
        self.assertEqual((output / 'best_ffield').read_text(), (self.root / 'native/ffield-1/ffield').read_text())
        self.assertEqual((self.root / 'ffield').read_text(), field_text())
        with self.assertRaisesRegex(ValueError, 'fresh campaign'):
            runner.run(self.request, sampler=lambda rows, count: {}, evaluator=evaluate)

    def test_completed_sample_survives_engine_failure(self):
        def evaluate(candidate):
            if candidate.name == 'ffield-1':
                raise RuntimeError('engine failed')
            output = candidate / 'jax_md_results'
            output.mkdir()
            (output / 'evaluation.json').write_text(json.dumps({'loss': 2, 'items': [{}]}))
        with self.assertRaises(RuntimeError):
            runner.run(self.request, sampler=lambda rows, count: {0: [1, 1], 1: [1, 1]}, evaluator=evaluate)
        self.assertEqual((self.root / 'native/short_results/results.csv').read_text().splitlines(), ['sample,loss', '0,2.0'])


if __name__ == '__main__':
    unittest.main()
