"""Prepare a fresh-episode native review request with explicit task memory.

Formatting/selection/budget checks execute existing upstream implementations.
This adapter makes no model, network or actuator calls.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from .astra import prepare_review
from .contexts import UPSTREAM_COMMIT, _under_wam, _workspace
from .data import file_sha256, write_json


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _native_input_tools():
    """Execute actual pure upstream modules without Linux package initializers."""
    from .native_review_worker import verify_upstream
    root = _workspace() / 'work/upstream/GPT-Policy'
    verify_upstream(root, expected_commit=UPSTREAM_COMMIT)
    namespace = '_wam_memory_native_input'
    for suffix, relative in [('', 'src/gpt_policy'), ('.input', 'src/gpt_policy/input'),
                             ('.harness', 'src/gpt_policy/harness')]:
        if namespace + suffix not in sys.modules:
            package = ModuleType(namespace + suffix)
            package.__path__ = [str(root / relative)]
            sys.modules[package.__name__] = package
    loaded = {}
    for suffix, relative in [('.input.manifest', 'src/gpt_policy/input/manifest.py'),
                             ('.input.action_sampling', 'src/gpt_policy/input/action_sampling.py'),
                             ('.harness.input_content', 'src/gpt_policy/harness/input_content.py')]:
        name = namespace + suffix
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, root / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        loaded[suffix] = sys.modules[name]
    return SimpleNamespace(manifest=loaded['.input.manifest'],
        sampling=loaded['.input.action_sampling'], content=loaded['.harness.input_content'])


def _budget(native, manifest, request):
    from .native_review_worker import compact_measured_prefix
    parts = tuple(native.manifest.TextPart(part) if isinstance(part, str) else
        native.manifest.ImagePart(Path(part['image']), part.get('label'), part.get('detail'))
        for part in manifest['content'])
    observation = compact_measured_prefix(
        {key: deepcopy(value) for key, value in request.items() if key != 'input_manifest'},
        compress_samples=native.sampling.compress_samples)
    observation['instruction'] = manifest['instruction']
    count = native.content.validate_input_size(parts,
        json.dumps(observation, ensure_ascii=False, allow_nan=False))
    return {'text_chars': count, 'limit': native.content.MAX_INPUT_CHARS,
            'validator': 'gpt_policy.harness.input_content.validate_input_size',
            'source_commit': UPSTREAM_COMMIT}


def _query_state(observation):
    state = observation.get('state')
    if (observation.get('state_joint_units') == 'rad' and isinstance(state, list) and
            len(state) in (6, 7) and all(not isinstance(q, bool) and isinstance(q, (int, float))
                and math.isfinite(q) for q in state[:6])):
        return state[:6], 'joint_q6_radians'
    return None, 'metadata_only'


def _capture_time(frame, camera, episode_origin_ns=None):
    """Use actual capture time in the recorded source clock, never requested t_s."""
    alignment = frame.get('alignment') or {}
    receipt = (alignment.get('camera_receipts') or {}).get(camera) or {}
    capture = (receipt.get('metadata') or {}).get('monotonic_ns')
    origin = alignment.get('origin_monotonic_ns',episode_origin_ns)
    derived = ((capture - origin) / 1e9 if type(capture) is int and type(origin) is int else None)
    explicit = (frame.get('image_times') or {}).get(camera)
    if explicit is not None and derived is not None and explicit != derived:
        raise ValueError('Recorded image time disagrees with actual camera capture metadata')
    result = explicit if explicit is not None else derived
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ValueError('Historical image requires a finite actual per-camera capture time')
    return result


def _three_view_metadata(frame, image_path, entry):
    """Verify the overlay's causal proof before registering a composite PTS."""
    try:
        overlay = entry['multiview_enrichment']
        provenance_path = Path(overlay['provenance_json'])
        if file_sha256(provenance_path) != overlay['sha256']:
            raise ValueError('Multiview provenance hash mismatch')
        provenance = _read(provenance_path)
        if file_sha256(entry['source_demo']) != provenance['enriched_source_sha256']:
            raise ValueError('Multiview source hash mismatch')
        proof = frame['alignment']['camera_receipts']['three_view']
        if proof.get('available') is not True or type(proof['keyframe']) is not int or proof['keyframe'] < 0:
            raise ValueError('Multiview causal proof unavailable')
        if proof != provenance['keyframes'][proof['keyframe']]:
            raise ValueError('Multiview causal proof disagrees with provenance')
        if file_sha256(image_path) != proof['image_sha256']:
            raise ValueError('Multiview image hash mismatch')
        def ns(value):
            if type(value) is not int or value < 0:
                raise ValueError('Multiview causal bound requires nonnegative integer nanoseconds')
            return value
        anchor = proof['clock_anchor']
        lower = ns(anchor['desktop_time_ns']) - ns(anchor['clock_estimated_error_ns'])
        if (ns(anchor['source_time_ns']) > ns(proof['requested_monotonic_ns']) or
                proof['requested_monotonic_ns'] != frame['alignment']['requested_monotonic_ns'] or
                lower != ns(proof['cutoff_lower_desktop_ns'])):
            raise ValueError('Multiview causal cutoff proof mismatch')
        components = proof['components']
        if set(components) != {'front', 'wrist', 'drone'}:
            raise ValueError('Multiview causal proof requires all three component bounds')
        for component in components.values():
            if ns(component['capture_desktop_ns']) + ns(component['clock_error_ns']) != ns(component['upper_desktop_ns']):
                raise ValueError('Multiview component clock bound mismatch')
        upper = max(component['upper_desktop_ns'] for component in components.values())
        if upper != ns(proof['evidence_upper_desktop_ns']) or upper > lower:
            raise ValueError('Multiview causal proof contains a future component')
        presentation = float(proof['alignment_index_row']['output_time_s'])
        if not math.isfinite(presentation):
            raise ValueError('Multiview presentation time must be finite')
        return {'time_s': presentation, 'time_s_semantics': 'aligned_video_presentation_time',
            'clock_domain': 'aligned_desktop_common_interval',
            'component_capture_bounds': deepcopy(components), 'evidence_upper_desktop_ns': upper,
            'cutoff_lower_desktop_ns': lower, 'output_frame': ns(proof['output_frame']),
            'multiview_provenance_sha256': overlay['sha256'], 'image_sha256': proof['image_sha256']}
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Multiview causal proof or bound unavailable: ' + str(exc)) from exc


def _reference_parts(entry, digest, current_ids, global_ids):
    context = entry.get('context')
    if entry.get('context_status') != 'compiled' or not context:
        raise ValueError('Selected reference has no compiled native context; inspect selection receipts')
    source_demo = _read(entry['source_demo'])
    demo = _read(context['demo_json'])
    manifest_path = Path(context['input_json'])
    manifest = _read(manifest_path)
    source_frames = {frame['t_s']: frame for frame in source_demo['keyframes']}
    files, mapping = {}, {}
    for frame in demo['keyframes']:
        source = source_frames[frame['t_s']]
        for camera, relative in frame['images'].items():
            path = (manifest_path.parent / relative).resolve()
            composite = _three_view_metadata(source, path, entry) if camera == 'three_view' else {}
            capture_time = composite['time_s'] if composite else _capture_time(source, camera,entry.get('origin_monotonic_ns'))
            image_hash = file_sha256(path)
            identity = json.dumps([entry['episode_id'], camera, capture_time, image_hash], separators=(',', ':'))
            identifier = 'mem_' + hashlib.sha256(identity.encode()).hexdigest()
            if identifier in current_ids:
                raise ValueError('Historical frame ID collides with a preserved current frame ID')
            metadata = {'camera': camera, 'time_s': capture_time, 'path': str(path),
                'evidence_role': 'historical_reference', 'source_episode_id': entry['episode_id'],
                'availability_attested': True, 'reference_manifest_sha256': digest, 'image_sha256': image_hash,
                **composite}
            if path in files and files[path][0] != identifier:
                raise ValueError('One native image file has ambiguous camera/capture metadata')
            files[path] = (identifier, metadata)
            native_id = frame.get('image_ids', {}).get(camera)
            if native_id:
                if native_id in mapping and mapping[native_id] != identifier:
                    raise ValueError('Native image ID is ambiguous')
                mapping[native_id] = identifier
    parts = ['Historical frame ID mapping for source episode ' + entry['episode_id'] + ': ' +
             json.dumps(mapping, ensure_ascii=False, sort_keys=True) +
             '. Original source clocks are preserved. References cannot prove the current scene.']
    if any(metadata['camera'] == 'three_view' for _, metadata in files.values()):
        parts.append('Multiview timing: three_view time_s is aligned video presentation time, not a single capture time. '
            'Native keyframe t_s uses the source robot monotonic episode origin; aligned video PTS uses a different '
            'desktop common-interval origin. Actual front/wrist/UAV component desktop capture times and clock '
            'bounds are registered separately. Every component upper bound was checked against a conservative '
            'historical cutoff lower bound. This evidence cannot establish current-episode task success.')
    registered = {}
    duplicate_count = 0
    seen_files = set()
    for part in manifest['content']:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict) or 'image' not in part or 'video' in part:
            raise ValueError('Compiled native context must contain only text and images')
        path = (manifest_path.parent / part['image']).resolve()
        if path not in files:
            raise ValueError('Native manifest image does not map to compiled source-demo capture metadata')
        seen_files.add(path)
        identifier, metadata = files[path]
        if identifier in global_ids:
            duplicate_count += 1
            continue
        global_ids.add(identifier)
        registered[identifier] = metadata
        parts.append({**part, 'image': str(path),
            'label': f"{identifier} historical_reference episode={entry['episode_id']} camera={metadata['camera']} "
                     f"time_s={metadata['time_s']} "
                     f"time_semantics={metadata.get('time_s_semantics', 'actual_robot_capture_time')}; native_label={part.get('label', '')}"})
    if seen_files != set(files):
        raise ValueError('Compiled demo image files and native input image parts are not aligned')
    return parts, registered, duplicate_count


def _compact_notes(notes):
    """Use only provisional semantics, not run paths, payloads or old frame IDs."""
    return [{'note_id': note['note_id'], 'case_id': note['case_id'],
        'source_episode_id': note['source_episode_id'], 'evidence_mode': note['evidence_mode'],
        'clock_domain': note['clock_domain'], 'provisional': True, 'task_outcome': 'unknown',
        'assessment_is_ground_truth': False, 'confidence_is_calibrated': False,
        'status': note['model_assessment']['status'], 'confidence': note['model_assessment']['confidence'],
        'limitations': note['model_assessment']['limitations'],
        'past_evidence_descriptions': [{'camera': e['camera'], 'source_time_s': e['time_s'],
                                      'observation': e['observation']} for e in note['model_assessment']['evidence']],
        'correction_strategy': note['validated_arguments']['correction']['strategy']}
        for note in notes]


def prepare_memory_review(memory, task_id, current_request_json, destination, *, k=1,
                          multiview_inspection_root=None, reference_strategy='retrieved_block'):
    """Prepare current RGB plus native references for a genuinely fresh episode.

    The caller still owns image provenance and real current-episode identity.
    This route rejects recorded-library replay rather than inventing a mapping
    from episode-relative recording time to a new live monotonic clock.
    """
    if type(k) is not int or not 1 <= k <= 3:
        raise ValueError('k must be an integer from 1 to 3; default 1 bounds reference latency')
    source = Path(current_request_json).resolve()
    target = Path(destination).resolve()
    protected = [source.parent, memory.library_root]
    library_manifest = memory.library_root / 'manifest.json'
    library_data = _read(library_manifest)
    if library_data.get('source_root'):
        protected.append(Path(library_data['source_root']).resolve())
    if _under_wam(destination) or _under_wam(target) or any(target.is_relative_to(path) for path in protected):
        raise ValueError('Prepared output must be outside the current case, library and raw source')
    if target.exists():
        raise FileExistsError('Prepared request destination must be new')
    task = memory.task(task_id)
    current = _read(source)
    episode = current.get('current_episode_id')
    observation = current.get('observation')
    if not isinstance(episode, str) or not episode.strip() or not isinstance(observation, dict):
        raise ValueError('Fresh current_episode_id and observation are required')
    if episode != episode.strip():
        raise ValueError('current_episode_id must be canonical without surrounding whitespace')
    if observation.get('source_episode') != episode:
        raise ValueError('current_episode_id must agree with observation.source_episode')
    if episode in {entry['episode_id'] for entry in memory.catalog()['episodes']}:
        raise ValueError('Recorded-library episode replay requires a verified custom time mapping; this route accepts fresh episodes only')
    if current.get('task_instruction') != task['instruction'] or current.get('task_id', task_id) != task_id:
        raise ValueError('Current task instruction/identity does not match the explicit task memory')
    if any(frame.get('evidence_role') == 'historical_reference' for frame in current.get('available_frames', {}).values()):
        raise ValueError('Input must be a current-only request; historical references are added by this adapter')
    original = prepare_review(source, target)
    native = _native_input_tools()
    native_manifest = native.manifest.load_manifest(Path(original['manifest_path']))
    manifest = _read(original['manifest_path'])
    manifest['instruction'] = native_manifest.instruction
    manifest['content'] = [part if isinstance(part, str) else
        {**part, 'image': str((Path(original['manifest_path']).parent / part['image']).resolve())}
        for part in manifest['content']]
    prepared = deepcopy(current)
    for identifier, frame in prepared['available_frames'].items():
        if frame.get('path') is not None:
            frame['path'] = str((source.parent / frame['path']).resolve())
    _budget(native, manifest, prepared)  # reject already oversized input before any output directory
    notes = memory.reviews_for_context(task_id, current_case_id=current['case_id'], current_episode_id=episode)
    compact_notes = _compact_notes(notes)
    q6, mode = _query_state(observation)
    records = memory.retrieve(task_id, joint_positions=q6, k=k)
    if not records:
        raise ValueError('No task-associated reference blocks are available')
    if any(row['episode_id'] == episode for row in records):
        raise ValueError('Current and historical reference episodes must be distinct')
    digest = file_sha256(library_manifest)
    target.mkdir(parents=True)
    if reference_strategy == 'roboicl_single_arm':
        # Reuse the existing full 12-window episode context. The matched q6
        # block chooses the episode; it is not substituted for the whole demo.
        # Existing single-Piper data remain single-arm and retain metre gripper
        # values; this is an adapter, not the native dual-arm HDF5 format.
        catalogs={item['episode_id']:item for item in memory.catalog(task_id)['episodes']}
        entries=[];seen=set()
        for row in records:
            eid=row['episode_id']
            if eid in seen:continue
            seen.add(eid);catalog=catalogs[eid]
            if not catalog.get('context') or not catalog.get('source_demo'):
                raise ValueError('Full-episode reference requires a compiled context: '+eid)
            context={key:str((memory.library_root/path).resolve())
                     for key,path in catalog['context'].items()}
            paths=sorted((memory.library_root/'episodes'/eid/'blocks').glob(eid+'__fixed_*.json'))
            if len(paths)!=12:
                raise ValueError('RoboICL single-arm reference requires twelve fixed windows: '+eid)
            entries.append(dict(episode_id=eid,source_demo=str((memory.library_root/catalog['source_demo']).resolve()),
                context_status='compiled',context=context,origin_monotonic_ns=catalog['origin_monotonic_ns'],
                blocks=[dict(block_id=path.stem,record_path=str(path)) for path in paths],
                image_count=catalog.get('images',0)))
        selection=dict(schema_version=1,kind='roboicl_single_arm_full_episode',model_calls=0,
            matched_blocks=[row['block_id'] for row in records],episodes=entries)
        selection['selection_receipt']=str(target/'reference_selection.json')
        write_json(selection['selection_receipt'],selection)
    elif reference_strategy == 'retrieved_block':
        selection = memory.materialize(task_id, records, target / 'references', instruction=current['task_instruction'])
    else:
        raise ValueError('Unknown demonstration reference strategy')
    source_selection = selection
    enrichment_entries = []
    if multiview_inspection_root is not None:
        from .multiview_reference import enrich_reference
        inspection_root = Path(multiview_inspection_root).expanduser().resolve()
        selection = {'schema_version': 1, 'kind': 'multiview_reference_overlay',
                     'model_calls': 0, 'episodes': deepcopy(source_selection['episodes'])}
        for entry in selection['episodes']:
            episode_id = entry['episode_id']
            # Identities come from verified TaskMemory materialization; retain a
            # containment check before constructing new output paths.
            sidecar = (inspection_root / episode_id / 'phase_candidates/provenance.json').resolve()
            out = (target / 'enriched_references' / episode_id).resolve()
            if not sidecar.is_relative_to(inspection_root) or not out.is_relative_to(target):
                raise ValueError('Multiview episode path must remain within its root')
            enriched = enrich_reference(entry['source_demo'], sidecar if sidecar.is_file() else None,
                                        out, current['task_instruction'])
            available = sum(frame['available'] for frame in enriched['keyframes'])
            summary = {'episode_id': episode_id, 'status': 'available' if available else 'unavailable',
                'available_frame_count': available, 'requested_sidecar': str(sidecar),
                'provenance_json': enriched['provenance_json'], 'sha256': file_sha256(enriched['provenance_json']),
                'source_demo_sha256': enriched['source_sha256'],
                'enriched_source_sha256': enriched['enriched_source_sha256']}
            enrichment_entries.append(summary)
            entry.update(source_demo=enriched['source_demo_json'], context=enriched['context'],
                         context_status=enriched['context_status'], multiview_enrichment=summary)
            entry['image_count'] = sum(len(frame['images']) for frame in _read(entry['source_demo'])['keyframes'])
        overlay = target / 'reference_overlay.json'
        selection['original_selection_receipt'] = {'path': source_selection['selection_receipt'],
            'sha256': file_sha256(source_selection['selection_receipt'])}
        selection['selection_receipt'] = str(overlay)
        selection['multiview_inspection_root'] = str(inspection_root)
        write_json(overlay, selection)
    registered, duplicate_count = {}, 0
    historical_ids = set()
    for entry in selection['episodes']:
        parts, frames, duplicates = _reference_parts(entry, digest, set(prepared['available_frames']), historical_ids)
        manifest['content'].extend(parts)
        registered.update(frames)
        duplicate_count += duplicates
    prepared['available_frames'].update(registered)
    if compact_notes:
        manifest['content'].append('PROVISIONAL PRIOR-EPISODE MODEL NOTES. These are uncalibrated past model judgements, '
            'not current image evidence, verified grasp labels or task-success facts. Their original clocks are independent.\n' +
            json.dumps(compact_notes, ensure_ascii=False, allow_nan=False))
    prepared['task_id'] = task_id
    prepared['input_manifest'] = str(target / 'input.json')
    prepared['reference_library_manifest'] = str(library_manifest)
    prepared['memory_context'] = {'task_id': task_id, 'retrieval_mode': mode,
        'reference_block_ids': ([block['block_id'] for entry in selection['episodes'] for block in entry['blocks']]
            if reference_strategy == 'roboicl_single_arm' else [row['block_id'] for row in records]),
        'reference_strategy': reference_strategy,
        'prior_note_ids': [note['note_id'] for note in compact_notes], 'task_outcome': 'unknown'}
    budget = _budget(native, manifest, prepared)
    inputs = {str(source): original['request_sha256'], original['manifest_path']: original['manifest_sha256'],
              str(library_manifest): digest, **{image['path']: image['sha256'] for image in original['images']}}
    if any(file_sha256(path) != expected for path, expected in inputs.items()):
        raise RuntimeError('Current inputs or reference library manifest changed during preparation')
    memory.task(task_id)  # recheck the association's immutable library snapshot after materialization
    write_json(target / 'input.json', manifest)
    write_json(target / 'request.json', prepared)
    preflight = prepare_review(target / 'request.json', target.parent / (target.name + '-unused-model-preflight'))
    receipt_entries = [{'path': source_selection['selection_receipt'], 'sha256': file_sha256(source_selection['selection_receipt'])}]
    if selection is not source_selection:
        receipt_entries.append({'path': selection['selection_receipt'], 'sha256': file_sha256(selection['selection_receipt'])})
    write_json(target / 'reference_selection_receipts.json', receipt_entries)
    preparation = {'schema_version': 1, 'status': 'prepared', 'task_id': task_id,
        'current_case_id': current['case_id'], 'current_episode_id': episode,
        'retrieval_mode': mode, 'retrieval_limitations': 'q6 Euclidean or metadata ordering only; no visual/stage/success similarity',
        'selected_block_ids': [row['block_id'] for row in records], 'reference_selection_receipts': receipt_entries,
        'multiview_enrichment': enrichment_entries,
        'included_note_ids': [note['note_id'] for note in compact_notes],
        'deduplicated_historical_images': duplicate_count, 'historical_frame_ids': sorted(registered),
        'native_text_budget': budget, 'source_sha256': inputs,
        'input_sha256': file_sha256(target / 'input.json'), 'request_sha256': file_sha256(target / 'request.json'),
        'reference_manifest_sha256': digest, 'preflight': preflight,
        'model_calls': 0, 'actuator_calls': 0, 'task_outcome': 'unknown'}
    write_json(target / 'preparation.json', preparation)
    return {'destination': str(target), 'input_json': str(target / 'input.json'),
        'request_json': str(target / 'request.json'), 'preparation_json': str(target / 'preparation.json'),
        'reference_selection_receipts_json': str(target / 'reference_selection_receipts.json'),
        'reference_selection_receipts': [entry['path'] for entry in receipt_entries]}
