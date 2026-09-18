#!/usr/bin/env python3
import itertools
from pathlib import Path
import random
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/probes'))
import grid_model_count_probe as probe


def valid(colours,rows,columns):
    return all(len({colours[i*columns+j] for i in (r,s) for j in (c,d)})>1
               for r,s in itertools.combinations(range(rows),2)
               for c,d in itertools.combinations(range(columns),2))


def satisfies(clauses,values):
    return all(any(values[abs(lit)-1]==(lit>0) for lit in clause) for clause in clauses)


class EncodingTest(unittest.TestCase):
    def test_exhaustive_colour_assignments_both_encodings(self):
        for rows,columns in ((1,1),(1,3),(2,2),(2,3)):
            for encoding in ('bits','onehot'):
                for anchor in (False,True):
                    _,clauses,factor=probe.encode(rows,columns,encoding,anchor)
                    total=0;unanchored=0
                    for colours in itertools.product(range(4),repeat=rows*columns):
                        values=[bool(c>>b&1) for c in colours for b in range(2)] if encoding=='bits' else [c==b for c in colours for b in range(4)]
                        want=valid(colours,rows,columns)
                        unanchored+=want
                        got=satisfies(clauses,values)
                        self.assertEqual(got,want and (not anchor or colours[0]==0))
                        total+=got
                    self.assertEqual(factor*total,unanchored)

    def test_onehot_rejects_illegal_assignments(self):
        _,clauses,_=probe.encode(1,1,'onehot')
        for values in itertools.product((False,True),repeat=4):
            self.assertEqual(satisfies(clauses,values),sum(values)==1)

    def test_random_larger_colours(self):
        rng=random.Random(501)
        for rows,columns in ((3,3),(4,5),(9,9)):
            _,clauses,_=probe.encode(rows,columns)
            for _ in range(100):
                colours=[rng.randrange(4) for _ in range(rows*columns)]
                values=[bool(c>>b&1) for c in colours for b in range(2)]
                self.assertEqual(satisfies(clauses,values),valid(colours,rows,columns))

    def test_nine_row_census(self):
        n,c,f=probe.encode(9,9)
        self.assertEqual((n,len(c),f),(162,5184,1))
        self.assertTrue(all(len(x)==8 for x in c))
        n,c,f=probe.encode(9,9,'onehot')
        self.assertEqual((n,len(c),f),(324,5751,1))

    def test_anchored_bits_primal_graph_is_complete(self):
        for r,c in ((2,2),(5,5),(6,6),(7,7),(9,9)):
            n,clauses,_=probe.encode(r,c,anchor=True)
            edges=set()
            for clause in clauses:
                # The anchor sets variables 1 and 2 false; drop satisfied
                # clauses, then strip those false positive literals.
                if -1 in clause or -2 in clause:continue
                remaining={abs(lit) for lit in clause if abs(lit)>2}
                edges.update(itertools.combinations(sorted(remaining),2))
            self.assertEqual(len(edges),(n-2)*(n-3)//2)

    def test_parse_never_certifies_timeout(self):
        self.assertEqual(probe.parse_count('# solutions\n252\n','original'),252)
        self.assertIsNone(probe.parse_count('TIMEOUT !\n# solutions\n252\n','original'))
        self.assertEqual(probe.parse_count('c s exact arb int 252\n','td'),252)
        self.assertEqual(probe.parse_count('c o CMD: timeout 1s flow_cutter\nc s exact arb int 63\n','td'),63)
        self.assertIsNone(probe.parse_count('c s exact double float 252\n','td'))
        self.assertIsNone(probe.parse_count('c s exact arb int 1\nc s exact arb int 2\n','td'))

    def test_invalid_geometry(self):
        for shape in ((0,2),(2,10),(-1,3)):
            with self.assertRaises(ValueError):probe.encode(*shape)

    def test_external_timeout_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            result=probe.bounded_run([sys.executable,'-c',
                'import time; print("c s exact arb int 4", flush=True); time.sleep(5)'],
                path,path/'timeout.log',.1,1)
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['exit_code'],0)
            self.assertLess(result['wall_seconds'],2)


if __name__=='__main__':unittest.main()
