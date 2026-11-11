"""Persistent explicit task associations and provisional native-model notes.

SQLite stores only an overlay. Retrieval distances remain scikit-learn results
from the existing library index; materialization remains native GPT-Policy.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .contexts import _under_wam
from .data import file_sha256


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value.strip()


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _review_provenance(request, validation):
    observation = request.get('observation') or {}
    source = observation.get('source_episode') if isinstance(observation, dict) else None
    source = source.strip() if isinstance(source, str) and source.strip() else None
    domain = request.get('clock_domain')
    domain = domain.strip() if isinstance(domain, str) and domain.strip() else 'unspecified'
    frames = request['available_frames']
    current_frames = {identifier: frame for identifier, frame in frames.items()
                      if frame.get('evidence_role') != 'historical_reference'}
    arguments = validation['arguments']
    cited = {entry['frame_id'] for entry in arguments['assessment']['evidence']}
    cited.update(identifier for step in arguments['correction']['steps'] for identifier in step['evidence_frame_ids'])
    provenance = {'source_episode_id': source, 'clock_domain': domain,
        'latest_input_frame_time_s': max((frame['time_s'] for frame in current_frames.values()), default=None),
        'latest_cited_evidence_time_s': max((current_frames[identifier]['time_s'] for identifier in cited
                                           if identifier in current_frames), default=None),
        'evidence_mode': validation['evidence_mode']}
    historical = []
    for identifier, frame in sorted(frames.items()):
        if frame.get('evidence_role') != 'historical_reference':
            continue
        reference_domain = frame.get('clock_domain')
        reference_domain = (reference_domain.strip() if isinstance(reference_domain, str)
                            and reference_domain.strip() else 'unspecified')
        historical.append({'id': identifier, 'source_episode_id': frame['source_episode_id'],
                           'clock_domain': reference_domain, 'time_s': frame['time_s']})
    if historical:
        provenance['historical_reference_times'] = historical
    return provenance


class TaskMemory:
    """A library-bound SQLite overlay; no model calls or outcome inference."""

    def __init__(self, library_root, memory_root):
        self.library_root = Path(library_root).expanduser().resolve()
        self.memory_root = Path(memory_root).expanduser().resolve()
        self.manifest_path = self.library_root / 'manifest.json'
        manifest_bytes = self.manifest_path.read_bytes()
        self._manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
        self._manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self._episodes = {e['episode_id']: e for e in self._manifest['episodes']}
        protected = [self.library_root]
        if self._manifest.get('source_root'):
            protected.append(Path(self._manifest['source_root']).resolve())
        if (_under_wam(memory_root) or _under_wam(self.memory_root) or
                any(self.memory_root.is_relative_to(path) for path in protected)):
            raise ValueError('Task memory must be outside the library and raw source data')
        self.memory_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.memory_root / 'task_memory.sqlite3'
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY, instruction TEXT NOT NULL, source TEXT NOT NULL,
                    associated_at TEXT NOT NULL, manifest_sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS task_episodes (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id), episode_id TEXT NOT NULL,
                    PRIMARY KEY(task_id, episode_id));
                CREATE TABLE IF NOT EXISTS review_notes (
                    note_id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(task_id),
                    case_id TEXT NOT NULL, model TEXT NOT NULL, thread_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL, note_json TEXT NOT NULL,
                    UNIQUE(model, thread_id, turn_id));
            ''')
            saved = dict(db.execute('SELECT key,value FROM metadata'))
            if saved and (saved.get('schema_version') != '1' or saved.get('library_root') != str(self.library_root)):
                raise ValueError('Task memory belongs to a different library or unsupported schema')
            db.executemany('INSERT OR IGNORE INTO metadata VALUES (?,?)',
                [('schema_version', '1'), ('library_root', str(self.library_root))])

    def _assert_manifest_snapshot(self, expected=None):
        try:
            actual = file_sha256(self.manifest_path)
        except OSError as exc:
            raise ValueError('Library manifest snapshot is unavailable') from exc
        if actual != self._manifest_sha256 or (expected is not None and expected != actual):
            raise ValueError('Library manifest snapshot drift; explicitly create a new association overlay for the changed library')

    @contextmanager
    def _db(self):
        if (self.database_path.is_symlink() or
                not self.database_path.resolve().is_relative_to(self.memory_root) or
                (self.database_path.exists() and self.database_path.stat().st_nlink > 1)):
            raise ValueError('Overlay database must remain outside source data and cannot alias another file')
        with closing(sqlite3.connect(self.database_path)) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA foreign_keys=ON')
            with db:
                yield db

    def associate_task(self, task_id, instruction, episode_ids, source='user'):
        """Add explicit episode memberships; never infer or overwrite a task."""
        task_id, instruction, source = (_text(task_id, 'task_id'), _text(instruction, 'instruction'), _text(source, 'source'))
        self._assert_manifest_snapshot()
        if isinstance(episode_ids, (str, bytes)):
            raise ValueError('episode_ids must be a collection of explicit episode IDs')
        episodes = sorted(set(_text(e, 'episode_id') for e in episode_ids))
        if not episodes or any(e not in self._episodes for e in episodes):
            raise ValueError('Every explicitly associated episode must exist in the library manifest')
        with self._db() as db:
            previous = db.execute('SELECT instruction,source,manifest_sha256 FROM tasks WHERE task_id=?', (task_id,)).fetchone()
            if previous is not None:
                self._assert_manifest_snapshot(previous['manifest_sha256'])
            if previous is not None and previous['instruction'] != instruction:
                raise ValueError('Existing task instruction cannot be silently relabeled; use a distinct task_id')
            if previous is not None and previous['source'] != source:
                raise ValueError('Existing association source cannot be silently relabeled; use a distinct task_id')
            db.execute('INSERT OR IGNORE INTO tasks VALUES (?,?,?,?,?)',
                (task_id, instruction, source, datetime.now(timezone.utc).isoformat(), self._manifest_sha256))
            db.executemany('INSERT OR IGNORE INTO task_episodes VALUES (?,?)', [(task_id, e) for e in episodes])
        return self.task(task_id)

    def task(self, task_id):
        task_id = _text(task_id, 'task_id')
        self._assert_manifest_snapshot()
        with self._db() as db:
            task = db.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
            if task is None:
                raise ValueError('Unknown explicitly associated task_id')
            result = dict(task)
            result['episode_ids'] = [row[0] for row in db.execute(
                'SELECT episode_id FROM task_episodes WHERE task_id=? ORDER BY episode_id', (task_id,))]
        self._assert_manifest_snapshot(result.get('manifest_sha256', ''))
        result.update(association_is_explicit=True, task_outcome='unknown', quality_filtering=False)
        return result

    def catalog(self, task_id=None):
        self._assert_manifest_snapshot()
        if task_id is None:
            with self._db() as db:
                for row in db.execute('SELECT DISTINCT manifest_sha256 FROM tasks'):
                    self._assert_manifest_snapshot(row[0])
        identifiers = self.task(task_id)['episode_ids'] if task_id is not None else sorted(self._episodes)
        return {'library_root': str(self.library_root), 'task_id': task_id,
                'episodes': [deepcopy(self._episodes[e]) for e in identifiers],
                'quality_filtering': False, 'task_outcome': 'unknown'}

    def retrieve(self, task_id, *, joint_positions=None, episode_id=None, at_s=None,
                 k=3, before_monotonic_ns=None):
        """Delegate ranking within each task episode, then merge real distances.

        Top k from every member episode contains the global task-subset top k.
        No global-index truncation, new distance, visual embedding or quality
        score is introduced. Raw monotonic cutoffs require one explicit episode.
        """
        from .retrieval import query
        task = self.task(task_id)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 1000:
            raise ValueError('k must be an integer between 1 and 1000')
        if episode_id is not None and episode_id not in task['episode_ids']:
            raise ValueError('Requested episode is not explicitly associated with this task')
        if before_monotonic_ns is not None and episode_id is None:
            raise ValueError('before_monotonic_ns requires an explicit episode_id; recording clocks differ')
        identifiers = [episode_id] if episode_id is not None else task['episode_ids']
        results = []
        for identifier in identifiers:
            options = dict(joint_positions=joint_positions, episode_id=identifier, at_s=at_s, k=k)
            if before_monotonic_ns is not None:
                options['before_monotonic_ns'] = before_monotonic_ns
            results.extend(query(self.library_root, **options))
        self._assert_manifest_snapshot(task['manifest_sha256'])
        if joint_positions is None:
            results.sort(key=lambda row: (row['episode_id'], row['start_s'], row['block_id']))
        else:
            results.sort(key=lambda row: (row['distance'], row['block_id']))
        return [{**row, 'task_id': task['task_id'], 'task_instruction': task['instruction'],
                 'task_association_source': task['source'], 'task_outcome': 'unknown'} for row in results[:k]]

    def materialize(self, task_id, records, destination, instruction=None):
        """Use actual RGB/NPZ selection and native demonstration compilation."""
        from .selection import export_selection
        task = self.task(task_id)
        rows = list(records)
        if not rows or any(row.get('episode_id') not in task['episode_ids'] for row in rows):
            raise ValueError('Every materialized record must belong to an explicitly associated task episode')
        return export_selection(self.library_root, rows, destination,
                                task['instruction'] if instruction is None else instruction)

    def append_review(self, run_dir, *, task_id=None):
        """Append a completed real native turn as a provisional episodic note.

        This loads/revalidates the native archive, never invokes a model, and
        never promotes a model assessment into a physical success annotation.
        Without an explicit task_id, unique exact instruction matching binds
        the archive to a previously user-associated task; no semantic guess.
        """
        from .astra import load_review_result
        root = Path(run_dir).resolve()
        required = ['native_receipt.json', 'request.json', 'reply.json', 'validation.json', 'usage.json', 'usage.jsonl']
        optional = ['native_decision.json', 'preflight.json', 'native_input.json',
                    'app_server_items.json', 'observation.json']
        if any(not (root / name).is_file() for name in required):
            raise RuntimeError('Native review archive is incomplete; no completed note can be appended')
        paths = {name: root / name for name in required + [n for n in optional if (root / n).is_file()]}
        before = {name: file_sha256(path) for name, path in paths.items()}
        result = load_review_result(root)
        request = json.loads(paths['request.json'].read_text(encoding='utf-8-sig'))
        receipt, validation = result['receipt'], result['validation']
        if (receipt.get('native_harness') != 'gpt_policy.harness.providers.codex.CodexSession' or
                receipt.get('native_sources_unchanged') is not True or receipt.get('decision_only') is not True or
                type(receipt.get('model_calls')) is not int or receipt['model_calls'] < 1 or
                receipt.get('actuator_calls') != 0):
            raise ValueError('Archive does not attest a completed real decision-only native model call')
        if receipt.get('case_id') != request['case_id'] or receipt.get('phase') != request['phase']:
            raise ValueError('Native receipt case/phase does not match its actual request')
        if 'native_decision.json' in paths:
            decision = json.loads(paths['native_decision.json'].read_text(encoding='utf-8-sig'))
            reply = json.loads(paths['reply.json'].read_text(encoding='utf-8-sig'))
            if decision.get('_wire') != reply or decision.get('arguments') != reply['arguments'] or decision.get('name') != reply['name']:
                raise ValueError('Native decision does not match the archived model reply')
        instruction = _text(request.get('task_instruction'), 'review task instruction')
        requested_task = request.get('task_id')
        if task_id is not None and requested_task is not None and task_id != requested_task:
            raise ValueError('Explicit task_id disagrees with the model request')
        chosen = task_id if task_id is not None else requested_task
        if chosen is None:
            with self._db() as db:
                matches = [r[0] for r in db.execute('SELECT task_id FROM tasks WHERE instruction=? ORDER BY task_id', (instruction,))]
            if len(matches) != 1:
                raise ValueError('Review task instruction must match one explicit task; supply task_id for ambiguity')
            chosen = matches[0]
        task = self.task(chosen)
        if task['instruction'] != instruction:
            raise ValueError('Review instruction does not match the explicitly associated task')
        if any(file_sha256(path) != before[name] for name, path in paths.items()):
            raise RuntimeError('Native review artifacts changed during memory validation')
        note_id = hashlib.sha256(_json([receipt['model'], receipt['thread_id'], receipt['turn_id']]).encode()).hexdigest()
        note = {'note_id': note_id, 'task_id': task['task_id'], 'case_id': request['case_id'],
            'phase': request['phase'], 'decision_time_s': request['decision_time_s'],
            'model': receipt['model'], 'thread_id': receipt['thread_id'], 'turn_id': receipt['turn_id'],
            'run_dir': str(root), 'provisional': True, 'note_kind': 'model_assessment',
            'assessment_is_ground_truth': False, 'confidence_is_calibrated': False,
            'task_outcome': 'unknown', 'cube_grasped': None, 'stable_lift': None,
            'physical_verification': None, 'model_assessment': deepcopy(validation['arguments']['assessment']),
            'validated_arguments': deepcopy(validation['arguments']),
            'source_artifacts': {name: {'path': str(path), 'sha256': before[name]} for name, path in paths.items()}}
        note.update(_review_provenance(request, validation))
        with self._db() as db:
            previous = db.execute('SELECT note_json FROM review_notes WHERE note_id=?', (note_id,)).fetchone()
            if previous is not None:
                old = json.loads(previous['note_json'])
                if old['task_id'] != task['task_id']:
                    raise ValueError('This native turn is already associated with a different task')
                return old
            db.execute('INSERT INTO review_notes VALUES (?,?,?,?,?,?,?)',
                (note_id, task['task_id'], note['case_id'], note['model'], note['thread_id'], note['turn_id'], _json(note)))
        return note

    def reviews(self, task_id):
        task = self.task(task_id)
        with self._db() as db:
            return [json.loads(row[0]) for row in db.execute(
                'SELECT note_json FROM review_notes WHERE task_id=? ORDER BY case_id,note_id', (task['task_id'],))]

    def reviews_for_context(self, task_id, *, current_case_id, current_episode_id):
        """Return hash-bound prior-episode notes, never same-episode hindsight.

        Legacy rows remain raw and readable through reviews(); absent episode
        provenance is not guessed or migrated. Different recording clocks are
        never compared to manufacture a causal ordering between episodes.
        """
        current_case_id = _text(current_case_id, 'current_case_id')
        current_episode_id = _text(current_episode_id, 'current_episode_id')
        required = {'source_episode_id', 'clock_domain', 'latest_input_frame_time_s',
                    'latest_cited_evidence_time_s', 'evidence_mode'}
        eligible = []
        for note in self.reviews(task_id):
            if not required.issubset(note) or not note.get('source_episode_id'):
                continue
            if note['case_id'] == current_case_id or note['source_episode_id'] == current_episode_id:
                continue
            artifacts = note.get('source_artifacts') or {}
            if not all(name in artifacts for name in ('request.json', 'validation.json')):
                raise RuntimeError('Context note lacks required source artifact hashes')
            try:
                for artifact in artifacts.values():
                    if file_sha256(artifact['path']) != artifact['sha256']:
                        raise RuntimeError('Context note source artifact hash mismatch')
                request = json.loads(Path(artifacts['request.json']['path']).read_text(encoding='utf-8-sig'))
                validation = json.loads(Path(artifacts['validation.json']['path']).read_text(encoding='utf-8-sig'))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError('Context note source artifacts are unavailable or malformed') from exc
            provenance = _review_provenance(request, validation)
            if (any(note.get(key) != value for key, value in provenance.items()) or
                    note['case_id'] != request['case_id'] or
                    note.get('validated_arguments') != validation['arguments'] or
                    note.get('model_assessment') != validation['arguments']['assessment']):
                raise RuntimeError('Context note provenance disagrees with its hash-bound source artifacts')
            eligible.append({**deepcopy(note), 'source_artifact_hashes_revalidated': True,
                'context_role': 'provisional_prior_episode_model_assessment',
                'context_scope': 'different_case_and_episode; no cross-clock timestamp ordering asserted'})
        return eligible
