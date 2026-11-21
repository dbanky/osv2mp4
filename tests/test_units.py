"""Tests that need no footage: metadata decoding, lens model, MP4 tagging."""
import subprocess
import sys

import numpy as np
import pytest

from osv2mp4.constants import FOV_HALF_DEG
from osv2mp4.djmd import parse_fields, read_varint
from osv2mp4.geometry import mei_shape
from osv2mp4.spherical import tag_spherical
from osv2mp4.stabilize import horizon_rot
from osv2mp4.verify import verify_output


def test_varint():
    assert read_varint(bytes([0x05]), 0) == (5, 1)
    assert read_varint(bytes([0xAC, 0x02]), 0) == (300, 2)


def test_parse_fields_nested_message():
    inner = bytes([0x08, 0x2A])                     # field 1, varint 42
    outer = bytes([0x12, len(inner)]) + inner       # field 2, length-delimited
    top = parse_fields(outer)
    assert parse_fields(top[2][0][1]) == {1: [(0, 42)]}


def test_parse_fields_survives_truncation():
    assert parse_fields(bytes([0x12, 0x10, 0x01])) is not None


def test_mei_radius_is_monotonic_to_the_rim():
    theta = np.radians(np.linspace(0.0, FOV_HALF_DEG, 500))
    r = mei_shape(theta)
    assert r[0] == 0.0
    assert np.all(np.diff(r) > 0)


def test_level_camera_gets_a_proper_rotation():
    R = horizon_rot([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


@pytest.fixture
def equirect_mp4(tmp_path):
    out = tmp_path / 'clip.mp4'
    r = subprocess.run(
        ['ffmpeg', '-v', 'error',
         '-f', 'lavfi', '-i', 'testsrc2=size=256x128:rate=30:duration=1',
         '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
         '-c:v', 'libx265', '-preset', 'ultrafast', '-x265-params', 'log-level=error',
         '-pix_fmt', 'yuv420p10le', '-tag:v', 'hvc1',
         '-c:a', 'aac', '-shortest', '-movflags', '+faststart', str(out)],
        capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f'ffmpeg cannot encode the test clip: {r.stderr[-200:]}')
    return str(out)


def test_untagged_file_fails_verify_then_passes_once_tagged(equirect_mp4):
    ok, problems = verify_output(equirect_mp4, quiet=True)
    assert not ok
    assert any('sv3d' in p or 'st3d' in p for p in problems)

    tag_spherical(equirect_mp4)
    ok, problems = verify_output(equirect_mp4, quiet=True)
    assert ok, problems


def test_cli_lists_the_commands():
    out = subprocess.run([sys.executable, '-m', 'osv2mp4', '--help'],
                         capture_output=True, text=True, check=True).stdout
    for cmd in ('convert', 'stitch', 'inspect', 'extract', 'cover', 'tag', 'verify', 'finalize'):
        assert cmd in out
