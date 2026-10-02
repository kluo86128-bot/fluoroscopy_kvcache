import contextlib
import errno
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiment.checkpoints import TopK
from experiment.data import question_text
from experiment.diagnostics import run_diagnostics
from experiment.io import is_storage_error, read_json, save_torch, write_error_status, write_json
from experiment.metrics import AnswerMonitor, test_prefix
from experiment.plots import percentage, render_comparison
from experiment.runner import run
from experiment.trainer import train
from test_experiment import Base, config


class BoundaryTests(Base):
    def test_raw_and_chat_keep_fixed_boundary_as_input(self):
        for kind in ('raw', 'chat'):
            cfg = config(question_format=kind)
            text = question_text('How many?\nAnswer:\n', 'Only answer', cfg['answer_boundary'])
            self.assertEqual(text.count('Answer:'), 1)
            self.assertTrue(text.endswith('Answer: '))
            ids = self.backend.encode_question(text, cfg)
            self.assertEqual(int(ids[0, -1]), 1 + ord(' ') % 30)

    def test_prefilled_sequence_probability_matches_incremental_forward(self):
        cfg = config()
        question = self.backend.encode_question(question_text('How many?', 'Only answer', cfg['answer_boundary']), cfg)
        private, _ = self.backend.student(self.initial, self.public_ids)
        from experiment.backend import join_cache
        monitor = AnswerMonitor(self.backend, self.observed, question, '12')
        result = monitor(private)
        cache = join_cache(private, self.observed)
        expected = 1.0
        for token in monitor.answer_ids[0]:
            logits, _ = self.backend.forward(ids=question, cache=cache)
            expected *= float(logits[0, -1].float().softmax(-1)[token])
            question = torch.cat((question, token.reshape(1, 1)), dim=1)
        self.assertAlmostEqual(result['answer_probability'], expected, places=8)
        self.assertEqual(result['answer_tokens'], 2)

    def test_newline_stops_decoding_and_is_not_in_matched_output(self):
        class OutputTokenizer:
            eos_token_id = 0
            def encode(self, text, add_special_tokens=False):
                return [1]
            def decode(self, values, skip_special_tokens=True):
                return ''.join({0: '', 1: '2', 2: '\n', 3: 'explanation'}[value] for value in values)
        self.backend.tokenizer = OutputTokenizer()
        next_tokens = iter((1, 2, 3))
        def forward(**kwargs):
            logits = torch.full((1, 1, 32), -100.0)
            logits[0, -1, next(next_tokens)] = 100
            return logits, self.observed
        with patch.object(self.backend, 'forward', side_effect=forward) as mock:
            generated = self.backend.rollout(self.observed, self.question_ids, 16, stop_strings=['\n'])
        self.assertEqual(generated.tolist(), [[1, 2]])
        self.assertEqual(mock.call_count, 2)
        with patch.object(self.backend, 'rollout', return_value=generated):
            result = test_prefix(self.backend, self.observed, self.public_ids, self.question_ids,
                                 self.initial, '2', [], 16, ['\n'])
        self.assertEqual(result['output'], '2')
        self.assertTrue(result['first_token_match'])
        self.assertTrue(result['full_match'])
        self.assertTrue(result['answer_match'])


class StorageTests(unittest.TestCase):
    def test_failed_save_keeps_previous_state_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            save_torch(path, {'value': torch.tensor([1])})
            before = path.read_bytes()
            def broken_save(value, output):
                output.write(b'incomplete')
                raise OSError(errno.ENOSPC, 'disk full')
            with patch('torch.save', side_effect=broken_save), self.assertRaises(OSError):
                save_torch(path, {'value': torch.tensor([2])})
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_json_failure_cleans_temporary_file_and_error_status_is_best_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.json'
            write_json(path, {'status': 'old'})
            with patch('experiment.io.os.replace', side_effect=OSError(errno.ENOSPC, 'full')), self.assertRaises(OSError):
                write_json(path, {'status': 'new'})
            self.assertEqual(read_json(path)['status'], 'old')
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            with patch('experiment.io.write_json', side_effect=OSError(errno.ENOSPC, 'full')), contextlib.redirect_stderr(io.StringIO()):
                write_error_status(path, {'status': 'failed'})

    def test_topk_metadata_only_and_last_resume_references_survive_pruning(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = TopK(directory, 2, 'base')
            prefix = torch.ones((1, 10, 16))
            pool.consider(1, prefix, 4)
            pool.consider(2, prefix, 3)
            with patch('torch.load', side_effect=AssertionError('metadata must not load payloads')):
                state = pool.state_dict()
            self.assertTrue(all('payload' not in row for row in state))
            pool.commit_resume()
            pool.consider(3, prefix * 3, 2)
            pool.consider(4, prefix * 4, 1)
            self.assertTrue((Path(directory) / 'step_00000001.pt').exists())
            restored = TopK(directory, 2, 'base')
            restored.load_state_dict(state)
            self.assertEqual([row['step'] for row in restored.entries], [2, 1])
            self.assertFalse((Path(directory) / 'step_00000004.pt').exists())
            restored.consider(3, prefix, 0)
            restored.commit_resume()
            self.assertFalse((Path(directory) / 'step_00000001.pt').exists())

    def test_storage_error_detects_chained_torch_exception(self):
        try:
            try:
                raise OSError(errno.ENOSPC, 'full')
            except OSError:
                raise RuntimeError('unexpected pos')
        except RuntimeError as error:
            self.assertTrue(is_storage_error(error))
        self.assertFalse(is_storage_error(RuntimeError('ordinary training error')))


class PlotTests(unittest.TestCase):
    def test_comparison_has_one_sequence_line_per_strategy_and_decimal_percent_ticks(self):
        from experiment.config import METHODS
        rows = [{'step': 0, 'answer_probability': 0.000001, 'first_token_probability': 0.8,
                 'support_score': None, 'ema_score': None},
                {'step': 400, 'answer_probability': 0.000002, 'first_token_probability': 0.9,
                 'support_score': 0.01, 'ema_score': 0.01}]
        series = {method: rows for method in METHODS}
        captured = []
        def capture(fig, path, **kwargs):
            captured.append((fig.axes[0], Path(path).name))
        with tempfile.TemporaryDirectory() as directory, patch('matplotlib.figure.Figure.savefig', autospec=True, side_effect=capture):
            render_comparison(directory, series)
        self.assertEqual(len(captured), 3)
        axis, name = captured[0]
        self.assertEqual(name, 'answer_probability_comparison.png')
        self.assertEqual(len(axis.lines), len(METHODS))
        self.assertEqual(axis.get_yscale(), 'linear')
        for actual, expected in zip(axis.lines[0].get_ydata(), [0.0001, 0.0002]):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(percentage(0.001), '0.001%')
        self.assertEqual(percentage(25), '25%')
        self.assertNotIn('e', percentage(1e-10))
        self.assertTrue(percentage(1e-10).endswith('%'))


class IntegrationTests(Base):
    def test_training_failure_is_not_masked_by_status_write_failure(self):
        from experiment.objectives import Objective
        original = Objective.compute
        calls = [0]
        def fail_after_initial(objective, prefix):
            calls[0] += 1
            if calls[0] > 1:
                raise RuntimeError('original training error')
            return original(objective, prefix)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch.object(Objective, 'compute', fail_after_initial), patch('experiment.io.write_json', side_effect=OSError(errno.ENOSPC, 'full')):
                with self.assertRaisesRegex(RuntimeError, 'original training error'):
                    train(self.backend, self.observed, self.public_ids, self.question_ids, self.initial,
                          'baseline', config(), directory, AnswerMonitor(self.backend, self.observed, self.question_ids, '12'))

    def test_disk_full_stops_runner_even_when_continue_on_error_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            data = root / 'tasks.json'
            write_json(data, [{'task_id': 'case', 'private_prefix': 'private', 'public_text': 'public', 'question': 'question', 'answer': '12'}])
            cfg = config(datasets=[str(data)], methods=['baseline', 'question_weighted_kv'], continue_on_error=True)
            with patch('experiment.runner.train', side_effect=OSError(errno.ENOSPC, 'full')) as mocked:
                with self.assertRaises(OSError):
                    run(cfg, root / 'run', backend=self.backend)
            self.assertEqual(mocked.call_count, 1)

    def test_re_evaluation_preserves_source_and_only_recomputes_saved_snapshots(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / 'tasks.json'
            write_json(data, [{'task_id': 'case', 'private_prefix': 'private', 'public_text': 'public',
                              'question': 'question', 'answer': '12', 'held_out_question': 'other question'}])
            cfg = config(datasets=[str(data)], methods=['baseline'], rounds=1, steps_per_round=3, save_topk=2)
            source = root / 'old_run'
            source.mkdir()
            run(cfg, source, backend=self.backend)
            original = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()}
            destination = root / 'new_diagnostics'
            self.assertEqual(run_diagnostics(cfg, destination, source, backend=self.backend), 0)
            self.assertEqual(original, {str(path.relative_to(source)): path.read_bytes() for path in source.rglob('*') if path.is_file()})
            method = destination / 'samples' / 'case' / 'baseline'
            history = [json.loads(line) for line in (method / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(history), 3)
            self.assertTrue(all(row['diagnostic_scope'] == 'saved_snapshots_only' for row in history))
            self.assertEqual(read_json(destination / 'test_summary.json')['baseline']['training_question']['prefix_tests'], 2)
            state = torch.load(source / 'samples' / 'case' / 'baseline' / 'latest.pt', weights_only=True)
            self.assertTrue(all('payload' not in row for row in state['topk']))

    def test_re_evaluation_reports_missing_rank_one_without_faking_top_one_rate(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = root / 'tasks.json'
            write_json(data, [{'task_id': 'case', 'private_prefix': 'private', 'public_text': 'public',
                              'question': 'question', 'answer': '12'}])
            cfg = config(datasets=[str(data)], methods=['baseline'], rounds=1, steps_per_round=2)
            source = root / 'old_run'
            source.mkdir()
            run(cfg, source, backend=self.backend)
            prefixes = source / 'samples' / 'case' / 'baseline' / 'prefixes'
            manifest = read_json(prefixes / 'manifest.json')
            (prefixes / manifest['prefixes'][0]['path']).unlink()
            destination = root / 'new_diagnostics'
            self.assertEqual(run_diagnostics(cfg, destination, source, backend=self.backend), 1)
            values = read_json(destination / 'test_summary.json')['baseline']['training_question']
            self.assertEqual(values['prefix_tests'], 1)
            self.assertIsNone(values['answer_match']['top1_rate'])
            self.assertEqual(len(read_json(destination / 'result.json')['failures']), 1)


if __name__ == '__main__':
    unittest.main()
