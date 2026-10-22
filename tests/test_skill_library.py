import tempfile,unittest
from pathlib import Path
from studio.skill_library import simulation_library


class SkillLibraryTests(unittest.TestCase):
    def test_missing_assets_are_visible_and_never_marked_qualified(self):
        with tempfile.TemporaryDirectory() as folder:
            rows=simulation_library(folder)
            self.assertTrue({'backflip','sideflip','upright_walk','handshake','jump','stairs','platform','obstacles'}<=set(r['id'] for r in rows))
            self.assertTrue(all(not r['available'] and r['blocker'] for r in rows))
            self.assertTrue(all(not r['continuous_switch'] and not r['hardware_qualified'] for r in rows))

    def test_primitive_is_distinct_from_trained_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'deployment/models/go2_isaac.xml';p.parent.mkdir(parents=True);p.touch()
            jump=next(r for r in simulation_library(folder) if r['id']=='jump')
            self.assertTrue(jump['available']);self.assertIn('无训练权重',jump['status'])


if __name__=='__main__':unittest.main()
