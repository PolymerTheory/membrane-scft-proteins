import importlib.util
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
spec = importlib.util.spec_from_file_location('demo_check', Path(__file__).resolve().parents[1] / 'examples/gpu-smoke/check_output.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

class DemoOutputTests(unittest.TestCase):
    def test_finite_shapes_and_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename, keys in {'wins.h5': ['W1','W2','Wp'], 'concentrations.h5': ['rhoA','rhoB','rhoH'], 'proteins.h5': ['prot1','prot2','prot3']}.items():
                with h5py.File(root / filename, 'w') as h:
                    for key in keys:
                        h[key] = np.zeros((4,32,32,32), dtype='f4')
            (root / 'solver.log').write_text('Done (0) (Time: 1s)\n')
            np.savetxt(root / 'FEs', np.zeros((4,2)))
            self.assertEqual(len(checker.verify(root)), 10)
            (root / 'solver.log').write_text('Setup (0)\n')
            with self.assertRaises(ValueError): checker.verify(root)
            (root / 'solver.log').write_text('Done (0) (Time: 1s)\n')
            with h5py.File(root / 'wins.h5', 'r+') as h:
                h['W1'][0,0,0,0] = np.nan
            with self.assertRaises(ValueError): checker.verify(root)
            with h5py.File(root / 'wins.h5', 'r+') as h:
                h['W1'][0,0,0,0] = 0
            np.savetxt(root / 'FEs', np.zeros((3,2)))
            with self.assertRaises(ValueError): checker.verify(root)
