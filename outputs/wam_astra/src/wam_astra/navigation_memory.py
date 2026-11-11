"""Recorded navigation evidence, never executable target or success labels."""
from __future__ import annotations
import bisect
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3


def displacement(start,end,robot):
    if not start or not end:return None
    keys=(['position_x','position_y','position_z','yaw'] if robot=='go2' else
          ['position.0','position.1','position.2','attitude.2'])
    try:
        a=[float(start[k]) for k in keys];b=[float(end[k]) for k in keys]
        if not all(math.isfinite(v) for v in a+b):return None
    except (KeyError,ValueError,TypeError):return None
    dx,dy,dz=[b[i]-a[i] for i in range(3)]
    c,s=math.cos(a[3]),math.sin(a[3])
    return dict(body_xyz_m=[c*dx+s*dy,-s*dx+c*dy,dz],
                yaw_delta_rad=math.atan2(math.sin(b[3]-a[3]),math.cos(b[3]-a[3])),
                distance_m=math.sqrt(dx*dx+dy*dy+dz*dz),
                semantics='measured_state_change_not_command',frame='own_robot_body_at_start',
                pose_reset_checked=False)


def preceding_rows(path,cutoffs):
    """Latest measured row whose clock upper bound precedes each cutoff.

    Full rows retain source row, robot state and recorded clock uncertainty.
    No camera-index nearest-neighbor row is assumed causal.
    """
    cutoffs=list(cutoffs)
    if cutoffs!=sorted(cutoffs):raise ValueError('Cutoffs must be sorted')
    result=[None]*len(cutoffs)
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                t=float(row['relative_time_s']);error=float(row['clock_estimated_error_ns'])/1e9
                if not math.isfinite(t) or not math.isfinite(error) or error<0:continue
            except (KeyError,ValueError):continue
            first=bisect.bisect_left(cutoffs,t+error)
            if first<len(result) and (result[first] is None or t>float(result[first]['relative_time_s'])):
                result[first]=row
    # A row eligible at an earlier cutoff remains eligible at every later one.
    for i in range(1,len(result)):
        previous=result[i-1]
        if previous and (result[i] is None or float(previous['relative_time_s'])>float(result[i]['relative_time_s'])):
            result[i]=previous
    return result


def select_candidates(windows):
    if not windows:return []
    selected={i for i in [0,len(windows)//2,len(windows)-1]}
    for robot in ('go2','uav'):
        for metric in ('distance_m','yaw_delta_rad'):
            eligible=[i for i,w in enumerate(windows) if w['motion'].get(robot)]
            eligible.sort(key=lambda i:(-abs(windows[i]['motion'][robot][metric]),i))
            selected.update(eligible[:2])
    return [windows[i] for i in sorted(selected)]


def rank_windows(rows,*,robot,exclude_episode,k=3):
    if robot not in ('go2','uav'):raise ValueError('Unknown robot')
    if not exclude_episode:raise ValueError('Explicit current episode ID required')
    if type(k) is not int or k<1:raise ValueError('k must be positive')
    candidates=[r for r in rows if r['episode_id']!=exclude_episode and r.get('motion',{}).get(robot)]
    candidates.sort(key=lambda r:(-r['motion'][robot]['distance_m'],r['window_id']))
    # One per historical episode first: avoid filling context with correlated windows.
    seen=set();diverse=[];remaining=[]
    for row in candidates:
        if row['episode_id'] in seen:remaining.append(row)
        else:diverse.append(row);seen.add(row['episode_id'])
    return (diverse+remaining)[:k]


def compact_reference(hit):
    """Keep controller-relevant evidence in context; full receipts stay in DB."""
    fields={
        'go2':['position_x','position_y','position_z','yaw','velocity_x','velocity_y','mode','error_code'],
        'uav':['position.0','position.1','position.2','attitude.2','armed','mode','odom_valid'],
        'piper':[*[f'joint_{j}.pos' for j in range(1,7)],'gripper.pos']}
    result={k:hit.get(k) for k in ('window_id','episode_id','date','motion','task_success','grasp_outcome','pi_handoff','scene_stage')}
    if 'visual_annotation' in hit:result['visual_annotation']=hit['visual_annotation']
    for endpoint in ('O0','O1'):
        observation=hit[endpoint]
        result[endpoint]={k:observation[k] for k in ('output_time_s','state_cutoff_relative_s','image_sha256')}
        result[endpoint]['measured']={}
        for robot,keys in fields.items():
            row=observation['measured'].get(robot)
            result[endpoint]['measured'][robot]=None if row is None else {
                k:row.get(k) for k in ['source_row','relative_time_s','clock_estimated_error_ns',*keys]}
    return result


def materialize(library,destination,*,robot,exclude_episode,k=3):
    root=Path(library).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    from .contexts import _under_wam
    if _under_wam(destination):raise ValueError('Cannot write raw data')
    with sqlite3.connect((root/'navigation.sqlite3').as_uri()+'?mode=ro',uri=True) as db:
        rows=[json.loads(r[0]) for r in db.execute('SELECT record_json FROM windows')]
    hits=rank_windows(rows,robot=robot,exclude_episode=exclude_episode,k=k)
    if not hits:raise ValueError('No independent historical navigation evidence')
    content=['HISTORICAL NAVIGATION EVIDENCE. These are measured outcomes from different episodes, not issued commands or proof of task success. Do not replay coordinates. Local robot odometry frames are independent. Position discontinuities have not been ruled out. States are the latest eligible PAST measurements at each endpoint and can be older than RGB. Use current images for the present task.']
    for hit in hits:
        annotation_path=Path(hit['O0']['image']).parent/'annotations.json'
        if annotation_path.is_file():
            annotations=json.loads(annotation_path.read_text(encoding='utf-8'))
            match=next((a for a in annotations['annotations'] if a['window_id']==hit['window_id']),None)
            if match:
                for endpoint in ('O0','O1'):
                    digest=hashlib.sha256(Path(hit[endpoint]['image']).read_bytes()).hexdigest()
                    if digest!=match['provenance'][endpoint]['image_sha256']:
                        raise ValueError('Visual annotation image hash mismatch')
                hit['visual_annotation']={k:v for k,v in match.items() if k not in ('provenance','window_id','episode_id')}
        content.append(json.dumps(compact_reference(hit),ensure_ascii=False,allow_nan=False))
        for endpoint in ('O0','O1'):
            # Relative POSIX paths work both in Windows and the native WSL worker.
            image=Path(os.path.relpath(hit[endpoint]['image'],destination)).as_posix()
            content.append(dict(image=image,label=hit['window_id']+' '+endpoint+' historical front/wrist/UAV',detail='high'))
    value=dict(instruction='Use historical navigation examples as context for current Astra positioning decisions; Pi performs grasping.',content=content)
    destination.mkdir(parents=True)
    (destination/'input.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    (destination/'selection.json').write_text(json.dumps(dict(robot=robot,excluded_episode=exclude_episode,
        rank_method='measured_distance_then_episode_diversity_not_visual_similarity',windows=[h['window_id'] for h in hits],
        task_outcomes_verified=False),indent=2),encoding='utf-8')
    return destination/'input.json'
