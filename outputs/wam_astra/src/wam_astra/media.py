"""Indexed, read-only MCAP decoding using mcap-python and Pillow."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from PIL import Image

from .data import file_sha256, read_json


def decode_raw_image(data):
    if len(data) < 4:
        raise ValueError('Truncated image metadata header')
    size = struct.unpack_from('<I', data)[0]
    if size > len(data)-4:
        raise ValueError('Truncated image metadata')
    meta = json.loads(data[4:4+size])
    if meta.get('pixel_format') != 'bgr8' or meta.get('dtype') != 'uint8':
        raise ValueError('Unsupported raw pixel layout')
    width, height, stride = (int(meta[k]) for k in ('width','height','stride'))
    payload = data[4+size:]
    if min(width,height) <= 0 or stride < width*3 or len(payload) != stride*height:
        raise ValueError('Invalid raw image dimensions or payload size')
    return meta, Image.frombytes('RGB', (width,height), payload, 'raw', 'BGR', stride)


def camera_catalog(robot_dir, summary=None):
    """Use recorder shard bounds; retain missing/unindexed shards as issues."""
    if not robot_dir:
        return {}, ['no_robot_camera_source']
    root = Path(robot_dir)/'raw'/'cameras'
    summary = summary or read_json(Path(robot_dir)/'summary.json')
    result, issues = {}, []
    for camera in ('front','wrist'):
        info = summary.get('cameras',{}).get(camera,{})
        shards = info.get('shards',[])
        if not shards:
            manifest = read_json(root/(camera+'.manifest.json'))
            shards = manifest.get('shards',[])
        selected = []
        for raw in shards:
            if not all(k in raw for k in ('first_monotonic_ns','last_monotonic_ns','first_wall_time_ns')):
                issues.append(camera+':shard_missing_timestamp_bounds');continue
            path = root/Path(raw.get('path','')).name
            if not path.is_file():
                issues.append(camera+':missing_shard:'+path.name);continue
            selected.append({**raw,'path':str(path)})
        selected.sort(key=lambda s:int(s['first_monotonic_ns']))
        result[camera] = selected
        catalogued = {Path(s['path']).name for s in selected}
        for path in sorted(root.glob(camera+'_*.mcap*')):
            if path.name not in catalogued:
                issues.append(camera+':unindexed_source_retained:'+path.name)
    return result, issues


def extract_frame(shards, query_ns, output, *, max_skew_ns=50_000_000, max_size=(480,360)):
    """Find the latest causal frame near host monotonic time.

    Recorder wall-clock bounds locate MCAP chunks. Actual embedded monotonic
    timestamps determine selection. Hashes cover the selected raw message and
    output image, not the entire shard. RGB thumbnails are derived copies.
    """
    from mcap.reader import make_reader
    output = Path(output)
    candidates = [s for s in shards if int(s['first_monotonic_ns'])-max_skew_ns <= query_ns
                  <= int(s['last_monotonic_ns'])+max_skew_ns]
    best, errors = None, []
    for shard in candidates:
        offset = int(shard['first_wall_time_ns'])-int(shard['first_monotonic_ns'])
        wall = int(query_ns)+offset
        try:
            with Path(shard['path']).open('rb') as stream:
                reader = make_reader(stream)
                for schema, channel, message in reader.iter_messages(starting_at=wall-max_skew_ns-50_000_000,
                                                                    ending_before=wall+max_skew_ns+50_000_001):
                    if channel.message_encoding != 'heterovla.raw_image.v1':
                        continue
                    size=struct.unpack_from('<I',message.data)[0]
                    meta=json.loads(message.data[4:4+size])
                    skew=int(meta['monotonic_ns'])-int(query_ns)
                    if -max_skew_ns <= skew <= 0 and (best is None or (abs(skew),int(meta['monotonic_ns'])) < best[0]):
                        best=((abs(skew),int(meta['monotonic_ns'])),meta,message.data,str(shard['path']),skew)
        except Exception as exc:
            errors.append({'path':str(shard['path']),'error':type(exc).__name__+':'+str(exc)})
    if best is None:
        return {'available':False,'requested_monotonic_ns':int(query_ns),
                'issue':'no_actual_frame_within_skew_bound','errors':errors}
    _,meta,payload,source,skew=best
    _,image=decode_raw_image(payload)
    image.thumbnail(max_size,Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True,exist_ok=True)
    image.save(output,format='PNG')
    return {'available':True,'path':str(output),'source_path':source,'metadata':meta,
            'requested_monotonic_ns':int(query_ns),'skew_ns':skew,
            'raw_message_sha256':hashlib.sha256(payload).hexdigest(),
            'image_sha256':file_sha256(output),'derived_size':list(image.size),'errors':errors}
