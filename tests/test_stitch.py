"""End-to-end stitch of a real clip. Needs footage, so it is opt-in:

    OSV2MP4_TEST_OSV=/path/to/clip.OSV pytest tests/test_stitch.py
"""
import os
import subprocess
import sys

import pytest

from osv2mp4.verify import verify_output

OSV = os.environ.get('OSV2MP4_TEST_OSV')


@pytest.mark.skipif(not OSV, reason='set OSV2MP4_TEST_OSV to a real .OSV file')
def test_one_second_stitch_is_fcp_ready(tmp_path):
    subprocess.run(
        [sys.executable, '-m', 'osv2mp4', 'convert', OSV, '-o', str(tmp_path),
         '--width', '1920', '--duration', '1', '--preset', 'ultrafast'],
        check=True)
    out = [p for p in os.listdir(tmp_path) if p.endswith('.mp4')]
    assert len(out) == 1
    ok, problems = verify_output(str(tmp_path / out[0]), quiet=True)
    assert ok, problems
