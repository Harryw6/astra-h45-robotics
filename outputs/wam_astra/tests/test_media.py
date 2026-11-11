import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'work/wam_demo_library/vendor'))


def fixture_mcap(path):
    from mcap.writer import Writer
    with path.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('heterovla.raw_image.v1', 'heterovla-layout-v1', b'{}')
        channel = writer.register_channel('/camera/front', 'heterovla.raw_image.v1', schema)
        for seq, t in enumerate([1_000_000_000, 1_100_000_000], 1):
            meta = {'camera':'front', 'monotonic_ns':t, 'wall_time_ns':t+10_000_000_000,
                    'width':2,'height':1,'stride':6,'pixel_format':'bgr8','dtype':'uint8','seq':seq}
            encoded=json.dumps(meta).encode()
            raw=bytes([0,0,255, 0,255,0])
            writer.add_message(channel,t+10_000_000_000,struct.pack('<I',len(encoded))+encoded+raw,t+10_000_000_000,seq)
        writer.finish()


def test_real_mcap_selects_nearest_actual_frame_and_preserves_bgr_and_source(tmp_path):
    from wam_astra.media import extract_frame
    source=tmp_path/'source.mcap';fixture_mcap(source)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    shard={'path':str(source),'first_monotonic_ns':1_000_000_000,'last_monotonic_ns':1_100_000_000,
           'first_wall_time_ns':11_000_000_000}
    result=extract_frame([shard],1_080_000_000,tmp_path/'frame.png',max_skew_ns=100_000_000)
    assert result['metadata']['monotonic_ns']==1_000_000_000
    assert result['skew_ns']==-80_000_000
    assert Image.open(tmp_path/'frame.png').getpixel((0,0))==(255,0,0)
    assert result['image_sha256']==hashlib.sha256((tmp_path/'frame.png').read_bytes()).hexdigest()
    assert before==hashlib.sha256(source.read_bytes()).hexdigest()


def test_missing_or_too_distant_frame_is_explicit_not_substituted(tmp_path):
    from wam_astra.media import extract_frame
    assert extract_frame([],1,tmp_path/'absent.png')['available'] is False
    source=tmp_path/'source.mcap';fixture_mcap(source)
    shard={'path':str(source),'first_monotonic_ns':1_000_000_000,'last_monotonic_ns':1_100_000_000,
           'first_wall_time_ns':11_000_000_000}
    result=extract_frame([shard],2_000_000_000,tmp_path/'absent.png',max_skew_ns=50_000_000)
    assert result['available'] is False
    assert not (tmp_path/'absent.png').exists()


def test_invalid_image_layout_is_rejected():
    from wam_astra.media import decode_raw_image
    with pytest.raises(ValueError):
        decode_raw_image(struct.pack('<I',1000)+b'{}')


def test_future_camera_frame_is_never_used_for_action_start(tmp_path):
    from wam_astra.media import extract_frame
    source=tmp_path/'source.mcap';fixture_mcap(source)
    shard={'path':str(source),'first_monotonic_ns':1_000_000_000,'last_monotonic_ns':1_100_000_000,
           'first_wall_time_ns':11_000_000_000}
    assert extract_frame([shard],980_000_000,tmp_path/'future.png')['available'] is False
    assert not (tmp_path/'future.png').exists()
