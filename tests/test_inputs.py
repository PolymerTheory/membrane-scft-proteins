"""Exercise public input switches through the CPU geometry executable."""
from pathlib import Path
import subprocess,tempfile,unittest,shutil,importlib.util
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[1]
class Inputs(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.work=Path(cls.tmp.name)
  shutil.copytree(ROOT/'src/tools',cls.work/'tools')
  cls.binary=cls.work/'protein_preview'
  subprocess.run(['c++','-O2','-DSCFT_STRN=4','-DPROTEIN_HEADER="protein_arcs.h"',str(ROOT/'src/protein_preview.cpp'),'-o',str(cls.binary)],check=True,capture_output=True)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def preview(self,cap,sym):
  d=self.work/f'{cap}_{sym}';d.mkdir(exist_ok=True)
  for p in (ROOT/'examples/geometry').iterdir():
   if p.is_file():(d/p.name).write_bytes(p.read_bytes())
  p=d/'prot_input.dat';p.write_text(p.read_text().replace('protein_cap_mode = 1',f'protein_cap_mode = {cap}').replace('protein_symmetrize = 1',f'protein_symmetrize = {sym}'))
  r=subprocess.run([str(self.binary),'input.dat'],cwd=d,capture_output=True)
  if r.returncode:return r,None
  with h5py.File(d/'proteins.h5') as h:a=np.stack([h[k][:] for k in ('prot1','prot2','prot3')])
  return r,a
 def test_geometry_switches(self):
  for cap in (0,1):
   fields=[]
   for sym in (0,1,2):
    r,a=self.preview(cap,sym);self.assertEqual(r.returncode,0,r.stderr.decode());self.assertTrue(np.isfinite(a).all());fields.append(a)
   a,add,copy=fields;n=a.shape[-1]
   np.testing.assert_allclose(add,a+a[...,(-np.arange(n))%n],rtol=1e-6,atol=1e-7)
   np.testing.assert_array_equal(copy[...,:n//2+1],a[...,:n//2+1])
   np.testing.assert_array_equal(copy[...,n//2+1:],a[...,n//2-1:0:-1])
 def test_geometry_demo_references(self):
  spec=importlib.util.spec_from_file_location('geometry_check',ROOT/'examples/check_geometry.py')
  checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
  for sym,example in [(0,'single-protein'),(1,'geometry')]:
   r,a=self.preview(1,sym);self.assertEqual(r.returncode,0)
   d=self.work/f'1_{sym}'
   self.assertEqual(len(checker.verify(d,ROOT/'examples'/example/'expected_output.json')),3)
   with h5py.File(d/'proteins.h5','r+') as h:h['prot1'][0,0,0,0]=float('nan')
   with self.assertRaises(ValueError):checker.verify(d,ROOT/'examples'/example/'expected_output.json')
 def test_unsupported_caps_rejected(self):
  r,a=self.preview(2,0);self.assertNotEqual(r.returncode,0);self.assertIn(b'protein_cap_mode must be',r.stdout+r.stderr)
if __name__=='__main__':unittest.main()
