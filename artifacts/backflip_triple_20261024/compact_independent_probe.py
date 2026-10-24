if __name__ == '__main__':
    import os,sys,runpy,json,hashlib,pathlib
    root=pathlib.Path('/root/gpufree-data/Rob2Nav');sys.path.insert(0,str(root));os.chdir(root)
    from rob2nav.deployment.core import SwitchController
    original=SwitchController.compact_entry_allowed
    def probe(self):
     if not self.hardware and self.compact_transitions and self.active is not None and self.pending is not None and self.active.atomic and self.active.digest()==self.pending.digest():return True
     return original(self)
    SwitchController.compact_entry_allowed=probe
    p=root/'outputs/backflip_triple_20261024'
    (p/('compact_independent_'+sys.argv[1]+'_contract.json')).write_text(json.dumps({'scope':'isolated simulation counterfactual; not deployment qualification','wrapper_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'change':'allow existing compact entry for identical atomic contract only; completion, measured stability, warmup and gain gates unchanged','seed':int(sys.argv[1]),'yaw':float(sys.argv[2])},indent=2))
    sys.argv=['verify_mujoco','--root',str(root),'--asset','source','--sequence',str(p/'sequence.json'),'--seed',sys.argv[1],'--initial-yaw-deg',sys.argv[2],'--max-seconds','150','--output',str(p/('compact_independent_'+sys.argv[1]))]
    runpy.run_module('rob2nav.deployment.verify_mujoco',run_name='__main__')
