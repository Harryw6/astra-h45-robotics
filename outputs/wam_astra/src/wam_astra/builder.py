"""Build a provenance-preserving Piper reference library outside the raw tree."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import (action_segment, compare_inventory, discover_episodes, ensure_separate_output,
                   file_sha256, fixed_windows, inventory, measured_action, observation_at, read_json,
                   read_piper_csv, receipt_record, save_signals, write_json)
from .media import camera_catalog, extract_frame


def _signal_summary(signal):
    if signal is None:
        return {'present':False,'rows':0}
    return {k:signal[k] for k in ('path','sha256','source_rows','invalid_rows','clock_reset_indices')} | {
        'present':True,'rows':len(signal['t_ns']),
        'time_span_ns':[int(signal['t_ns'][0]),int(signal['t_ns'][-1])] if len(signal['t_ns']) else None}


def _read_signal(path, *, command=False):
    try:
        return read_piper_csv(path,command=command),None
    except (ValueError,UnicodeError) as exc:
        return None,{'path':str(path),'sha256':file_sha256(path),
                     'parse_error':type(exc).__name__+':'+str(exc)}


def build_library(source_root, output, *, extract_images=True, compile_contexts=True,
                  dense_stride_s=2.0, progress=None):
    source_root=Path(source_root).resolve()
    output=ensure_separate_output(source_root,output)
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    if not np.isfinite(dense_stride_s) or dense_stride_s <= 0:
        raise ValueError('dense_stride_s must be finite and positive')
    before=inventory(source_root)
    output.mkdir(parents=True)
    write_json(output/'source_inventory_before.json',before)
    episodes=discover_episodes(source_root)
    manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),
              'source_root':str(source_root),'all_source_episodes_included':True,
              'source_files':len(before),'source_bytes':sum(r['size'] for r in before),
              'organization':'RoboICL-inspired O0,A,receipt,O1; native RoboDojo HDF5 protocol not asserted',
              'fixed_windows_per_episode':12,'window_s':1,'dense_stride_s':dense_stride_s,
              'eligibility_policy':'All episodes retained; original quality flags are provenance only.',
              'task_success_labels':'Unknown unless explicitly sourced; no inferred grasp labels.',
              'episodes':[],'block_count':0,'image_count':0,'compiled_context_count':0}
    parsed_hashes=[]
    with (output/'blocks.jsonl').open('w',encoding='utf-8') as index:
        for number,episode in enumerate(episodes,1):
            eid=episode['episode_id'];directory=output/'episodes'/eid;directory.mkdir(parents=True)
            if progress:progress(f'[{number}/{len(episodes)}] {eid}')
            robot=Path(episode['robot_dir']) if episode['robot_dir'] else None
            meta=read_json(robot/'meta.json') if robot else {}
            summary=read_json(robot/'summary.json') if robot else {}
            state,state_error=_read_signal(robot/'raw'/'piper_state.csv') if robot else (None,None)
            commands,command_error=_read_signal(robot/'raw'/'piper_gamepad.csv',command=True) if robot else (None,None)
            for signal in (state,commands):
                if signal:parsed_hashes.append({'path':signal['path'],'sha256':signal['sha256']})
            metadata_files=[]
            for parent in [robot,Path(episode['drone_dir']) if episode['drone_dir'] else None,
                           Path(episode['joint_dir']) if episode['joint_dir'] else None]:
                if parent:
                    metadata_files.extend(p for p in parent.glob('*.json') if p.is_file())
            if episode['alignment_path']:metadata_files.append(Path(episode['alignment_path']))
            if robot:metadata_files.extend((robot/'raw'/'cameras').glob('*.manifest.json'))
            for path in metadata_files:parsed_hashes.append({'path':str(path),'sha256':file_sha256(path)})
            catalog,issues=camera_catalog(robot,summary)
            for error in (state_error,command_error):
                if error:
                    parsed_hashes.append({'path':error['path'],'sha256':error['sha256']})
                    issues.append('parse_error:'+error['path']+':'+error['parse_error'])
            row={**episode,'quality_ok':summary.get('quality_ok'),'task':meta.get('task'),
                 'instruction':meta.get('instruction'),'task_label_verified':False,
                 'task_success':meta.get('task_success',meta.get('success')),
                 'state':_signal_summary(state),'commands':_signal_summary(commands),
                 'camera_shards':catalog,'issues':issues,'fixed_blocks':0,'dense_blocks':0,
                 'images':0,'context':None,'signals_path':f'episodes/{eid}/signals.npz'}
            write_json(directory/'source_metadata.json',{'robot_meta':meta,'robot_summary':summary,
                       'alignment':read_json(episode['alignment_path']) if episode['alignment_path'] else None,
                       'other_metadata':[{'path':str(p),'value':read_json(p)} for p in metadata_files if p not in
                                         ([robot/'meta.json',robot/'summary.json'] if robot else [])]})
            save_signals(directory/'signals.npz',state,commands)
            if not state or len(state['t_ns'])<2:
                row['issues'].append('no_usable_piper_state_span');write_json(directory/'episode.json',row)
                manifest['episodes'].append(row);continue
            if state['clock_reset_indices']:
                row['issues'].append('non_monotonic_state_preserved_in_npz; no_ambiguous_windows_indexed')
                write_json(directory/'episode.json',row);manifest['episodes'].append(row);continue
            if commands and commands['clock_reset_indices']:
                row['issues'].append('non_monotonic_command_time_preserved; command_segments_unavailable')
                usable_commands=None
            else:usable_commands=commands
            start,end=int(state['t_ns'][0]),int(state['t_ns'][-1])
            row['origin_monotonic_ns']=start
            windows=[('fixed',a,b) for a,b in fixed_windows(start,end)]
            stride=max(1,int(dense_stride_s*1e9))
            windows += [('dense',t,t+1_000_000_000) for t in range(start,end-1_000_000_000+1,stride)]
            keyframes={}
            for kind,a,b in windows:
                block_id=f'{eid}__{kind}_{a-start:015d}'
                O0,O1=observation_at(state,a),observation_at(state,b)
                action=action_segment(usable_commands,a,b)
                lo,hi=np.searchsorted(state['t_ns'],[a,b],side='left')
                receipt_end=int(np.searchsorted(state['t_ns'],b,side='right'))
                receipt=receipt_record(command_rows=action['command_sent_rows'],
                    measured_rows=state['source_row'][lo:receipt_end].tolist(),tracking=None)
                receipt['tracking_unavailable_reason']='Recorded target units not independently verified; no command/feedback equivalence assumed.'
                events=[]
                if O0['piper'] is None or O1['piper'] is None:events.append('missing_or_stale_state_endpoint')
                if hi>lo and np.any(np.diff(state['t_ns'][lo:min(hi+1,len(state['t_ns']))])>250_000_000):
                    events.append('recorded_state_gap_over_250ms')
                if hi>lo+1 and np.any(np.diff(state['g'][lo:hi][np.isfinite(state['g'][lo:hi])]) != 0):
                    events.append('recorded_gripper_value_change')
                record={'schema_version':1,'block_id':block_id,'episode_id':eid,'kind':kind,
                        'start_ns':a,'end_ns':b,'start_s':(a-start)/1e9,'end_s':(b-start)/1e9,
                        'O0':O0,'A':action,'receipt':receipt,'O1':O1,'events':events,
                        'numeric_reference_semantics':'measured_states; recorded targets require unit verification',
                        'source_quality_ok':summary.get('quality_ok'),'camera_locator':{'front':a,'wrist':a}}
                if kind=='fixed':
                    for t,observation in ((a,O0),(b,O1)):
                        if t not in keyframes:
                            images,frame_receipts={},{}
                            if extract_images:
                                for camera,shards in catalog.items():
                                    image_path=directory/'images'/f'{t-start:015d}_{camera}.png'
                                    result=extract_frame(shards,t,image_path)
                                    if result['available']:
                                        result['path']=image_path.relative_to(output).as_posix()
                                        images[camera]=f'../../episodes/{eid}/images/{image_path.name}';row['images']+=1
                                    frame_receipts[camera]=result
                            frame={'t_s':(t-start)/1e9,'images':images,
                                   'state':{'piper':observation['piper']} if observation['piper'] else {},
                                   'alignment':{'requested_monotonic_ns':t,'camera_receipts':frame_receipts,
                                                'state_source_row':observation.get('source_row')}}
                            keyframes[t]=frame
                        observation['images']=keyframes[t]['alignment']['camera_receipts']
                    keyframes[a]['action']=measured_action(state,a,b,start)
                record_path=f'episodes/{eid}/blocks/{block_id}.json'
                write_json(output/record_path,record)
                p=O0['piper'] or {}
                indexed={'block_id':block_id,'episode_id':eid,'date':episode['date'],'start_ns':a,'end_ns':b,
                         'start_s':(a-start)/1e9,'end_s':(b-start)/1e9,'kind':kind,
                         'q_start':p.get('joint_measured_rad'),'gripper_start_m':p.get('gripper_measured_m'),
                         'events':events,'record_path':record_path}
                index.write(json.dumps(indexed,ensure_ascii=False,allow_nan=False)+'\n')
                row[kind+'_blocks']+=1;manifest['block_count']+=1
            valid_frames=[keyframes[t] for t in sorted(keyframes) if keyframes[t]['images']]
            if valid_frames:
                source_demo=output/'source_demos'/eid/'demo.json'
                demo={'schema_version':1,'title':eid,'source':str(robot),'demonstrator':'recorded_human_teleoperation',
                      'summary':'Historical Piper measurements and real RGB frames. Task outcome is unverified.',
                      'outcome':'unknown','source_robot':{'model':'AgileX Piper','bimanual':False},
                      'coordinate_frame':'joint_angles_rad; gripper_measured_m; shared_host_monotonic_time',
                      'numeric_reference_semantics':'measured_states','keyframes':valid_frames}
                write_json(source_demo,demo)
                row['source_demo']=source_demo.relative_to(output).as_posix()
                if compile_contexts:
                    from .contexts import compile_demo
                    result=compile_demo(source_demo,output/'contexts'/eid,
                        'Review the proposed Piper actions against current observations and these historical demonstrations. '
                        'Identify visible task deviations. Historical measured trajectories are references, not issued controls. '
                        'The demonstration task outcome is unverified. Do not infer actuator acknowledgement or grasp success.')
                    row['context']={k:str(Path(v).relative_to(output)) for k,v in result.items()}
                    manifest['compiled_context_count']+=1
            else:row['issues'].append('no_actual_reference_images; numeric_records_retained')
            manifest['image_count']+=row['images']
            write_json(directory/'episode.json',row);manifest['episodes'].append(row)
            if progress:progress(f'  blocks={row["fixed_blocks"]+row["dense_blocks"]}, images={row["images"]}, native_context={bool(row["context"])}')
    write_json(output/'parsed_file_hashes.json',parsed_hashes)
    after=inventory(source_root)
    write_json(output/'source_inventory_after.json',after)
    verification=compare_inventory(before,after)
    verification['parsed_hashes_checked']=len(parsed_hashes)
    verification['parsed_hash_mismatches']=[r['path'] for r in parsed_hashes if file_sha256(r['path'])!=r['sha256']]
    verification['unchanged']=verification['unchanged'] and not verification['parsed_hash_mismatches']
    manifest['source_verification']=verification
    write_json(output/'manifest.json',manifest)
    if not verification['unchanged']:raise RuntimeError('Source changed during build; inspect source_verification')
    return {'manifest':str(output/'manifest.json'),'blocks':manifest['block_count'],
            'images':manifest['image_count'],'contexts':manifest['compiled_context_count']}
