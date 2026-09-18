#!/usr/bin/env python3
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/probes'))
from cut_join_features import summarize


class JoinFeaturesTest(unittest.TestCase):
    def test_small_orbit_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); inp=root/'input'; output=root/'features'
            inp.write_bytes(struct.pack('<8sIQQQ',b'R8SQT01\0',8,1,0,3))
            run=subprocess.run([str(ROOT/'build/cut_join_features'),str(inp),'10'],
                               capture_output=True,text=True,check=True,timeout=20)
            output.write_text(run.stdout)
            result=summarize(output)
            self.assertEqual(result['completion']['records'],1)
            # Empty selected halves contribute one 1x1 tile; the full active
            # complement has no valid binary colouring and contributes none.
            self.assertEqual(result['totals']['tiles'],1)
            self.assertEqual(result['tile_occupancy'],1/128)
            lines=run.stdout.splitlines(keepends=True)
            for bad in (lines[:-1], lines+lines[-1:], lines[1:]):
                output.write_text(''.join(bad))
                with self.assertRaises(ValueError):summarize(output)

    def test_research_headers_are_self_contained(self):
        for name in ('response_model','cut_geometry','cut_reference_model',
                     'cut_tile_index','cut_histogram_model','cut_export'):
            with self.subTest(header=name):
                text=f'#include "{name}.hpp"\n#include "{name}.hpp"\nint main() {{}}\n'
                run=subprocess.run(['g++','-std=c++17','-fsyntax-only','-x','c++','-',
                                    '-I',str(ROOT/'research/probes')],
                                   input=text,capture_output=True,text=True,timeout=30)
                self.assertEqual(run.returncode,0,run.stderr)


if __name__=='__main__':unittest.main()
