"""Explicit SEA ray geometry for isolated native-layer validation, not navigation."""
import hashlib
import importlib.util
import math
from pathlib import Path

SOURCE_SHA256='92bc25dcc8afd2a0c02fb0d62a30ebee22c80d55b5e3e208f7ee7b0c114e1ea6'
SOURCE_COMMIT='560de73c094a750a6a358bc1920ebc9ef1d9f3c5'


def ray_angles(num_rays=41,fov_deg=240.):
    if isinstance(num_rays,bool) or not isinstance(num_rays,int) or num_rays<3 or num_rays%2!=1:
        raise ValueError('An odd ray count >=3 is required')
    if not math.isfinite(fov_deg) or not 0<fov_deg<=360:raise ValueError('Invalid FOV')
    return [math.radians(-fov_deg/2+i*fov_deg/(num_rays-1)) for i in range(num_rays)]


def load_native_cbf(source_file,num_rays=41,fov_deg=240.):
    """Load only the audited source and pass FOV explicitly, on CPU by default.

    This does not patch upstream actors, checkpoints or a running robot. Loading
    state_dict can overwrite ray buffers; validate_geometry must run afterwards.
    """
    ray_angles(num_rays,fov_deg)
    path=Path(source_file).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA256:raise ValueError('SEA CBF source hash mismatch')
    spec=importlib.util.spec_from_file_location('_rob2nav_audited_sea_cbf',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    layer=module.ExactLSECBFLayer(num_rays=num_rays,fov_deg=fov_deg)
    validate_geometry(layer,num_rays,fov_deg)
    return layer


def validate_geometry(layer,num_rays=41,fov_deg=240.):
    import torch
    angles=torch.tensor(ray_angles(num_rays,fov_deg),device=layer.ray_unit_vectors.device)
    expected=torch.stack((angles.cos(),angles.sin()),-1)
    if layer.ray_unit_vectors.shape!=expected.shape or not torch.allclose(layer.ray_unit_vectors,expected,atol=1e-6,rtol=0):
        raise ValueError('Ray angles do not match observation FOV; check loaded buffers')
