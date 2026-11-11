"""Read-only bounded PCD preview: ASCII / binary XYZ, not binary_compressed."""
import hashlib
from pathlib import Path
import numpy as np


def projection(path, z_min=-float('inf'), z_max=float('inf'), max_points=20000):
    path = Path(path)
    with path.open('rb') as stream:
        header = {}; header_bytes = 0
        while True:
            line = stream.readline(); header_bytes += len(line)
            if not line or header_bytes > 65536: raise ValueError('无效 PCD header')
            words = line.decode('ascii').strip().split()
            if not words or words[0].startswith('#'): continue
            header[words[0].upper()] = words[1:]
            if words[0].upper() == 'DATA': break
        mode = header['DATA'][0]; names = header.get('FIELDS', header.get('FIELD'))
        if not names or not {'x', 'y', 'z'}.issubset(names): raise ValueError('PCD 需要 x/y/z 字段')
        count = int(header.get('POINTS', [str(int(header['WIDTH'][0])*int(header.get('HEIGHT', ['1'])[0]))])[0])
        if not 0 < count <= 200000000: raise ValueError('点数不在支持范围')
        stride = max(1, (count+max_points-1)//max_points)
        if mode == 'ascii':
            xyz = []; ix, iy, iz = [sum(map(int, header.get('COUNT', ['1']*len(names))[:names.index(n)])) for n in ('x', 'y', 'z')]
            for index, line in enumerate(stream):
                if index >= count: break
                if index % stride: continue
                values = line.split(); xyz.append([float(values[ix]), float(values[iy]), float(values[iz])])
            points = np.asarray(xyz, dtype=float).reshape(-1, 3)
        elif mode == 'binary':
            size = list(map(int, header['SIZE'])); types = header['TYPE']; counts = list(map(int, header.get('COUNT', ['1']*len(names))))
            if not len(names) == len(size) == len(types) == len(counts): raise ValueError('非法 PCD schema')
            dtype = []
            for name, width, kind, n in zip(names, size, types, counts):
                if kind not in ('F', 'I', 'U') or width not in (1, 2, 4, 8) or not 1 <= n <= 64:
                    raise ValueError('不支持的 PCD 字段')
                base = np.dtype('<'+{'F': 'f', 'I': 'i', 'U': 'u'}[kind]+str(width))
                dtype.append((name, base) if n == 1 else (name, base, (n,)))
            dtype = np.dtype(dtype)
            if path.stat().st_size-header_bytes < count*dtype.itemsize: raise ValueError('PCD 被截断')
            mapped = np.memmap(path, mode='r', dtype=dtype, offset=header_bytes, shape=(count,))
            points = np.column_stack([mapped[n][::stride] for n in ('x', 'y', 'z')]).astype(float)
            del mapped
        else: raise ValueError('暂不支持 binary_compressed；请用 PCL 转为 binary 或 ASCII 后预览，原地图保留')
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    return dict(points=points.tolist(), source_points=count, preview_points=len(points),
                z_min=z_min if np.isfinite(z_min) else None, z_max=z_max if np.isfinite(z_max) else None)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()
