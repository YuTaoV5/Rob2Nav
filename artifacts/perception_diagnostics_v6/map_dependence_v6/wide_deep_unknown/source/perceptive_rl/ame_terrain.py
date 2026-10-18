"""Independent metric implementation of SII AME stakes1/stakes2 geometry.

8 m tile, 5 cm grid, 2 m centre platform, -2 m pit. Seeded heights replace
upstream's unseeded RNG. No source code or model weights copied from AME.
The 12 cm profile is an extra stress test, not a paper-layout reproduction.
"""
import numpy as np

KINDS = ("ame_stakes", "ame_alternate", "ame_stakes12")


def parameters(difficulty, kind):
    if kind not in KINDS or not 0 <= difficulty <= 1:
        raise ValueError((kind, difficulty))
    alternate = kind == "ame_alternate"
    return {"side": .4-difficulty*(.4-.2) if kind != "ame_stakes12" else .12,
            "gap": .05+difficulty*(.15-.05) if alternate else .1+difficulty*(.3-.1),
            "column_gap": .2*(1-difficulty) if alternate else .1,
            "pit": -2., "platform": 2., "height_max": .03,
            "grid": .01 if kind == "ame_stakes12" else .05,
            "seed": 42, "tile": [8., 8.],
            "origin": [4., 4., 0.], "goal_x": 3.5}


def heightfield(difficulty, kind):
    p = parameters(difficulty, kind)
    scale = p["grid"]; n = round(8/scale)
    # Match upstream int truncation, including its floating-point boundary cases.
    side = max(1, int(p["side"]/scale))
    gap = max(0, int(p["gap"]/scale))
    sep = max(0, int(p["column_gap"]/scale))
    low, high = side//2, side-side//2
    rng = np.random.default_rng(p["seed"])
    heights = np.full((n,n), -2.)
    for axis in (0,1):
        for index, pos in enumerate(range(0,n,side+gap)):
            offsets = ((-1 if index % 2 else 1)*(sep//2),) if kind == "ame_alternate" else (-max((side+sep)//2,low), max((side+sep)//2,low))
            for offset in offsets:
                transverse = int(np.clip(n//2+offset, low, n-high))
                x,y = (pos,transverse) if axis == 0 else (transverse,pos)
                heights[max(0,x-low):min(n,x+high), max(0,y-low):min(n,y+high)] = rng.integers(-6,7)*.005
    half = round(p["platform"]/(2*scale))
    heights[n//2-half:n//2+half,n//2-half:n//2+half] = 0.
    return heights


def boxes(difficulty, kind):
    """Merge identical row spans into boxes; exact vertical walls in both engines."""
    h = heightfield(difficulty,kind); scale = parameters(difficulty,kind)["grid"]
    result = [(0.,8.,0.,8.,-2.,-2.2)]
    active = {}
    for ix in range(h.shape[0]+1):
        spans = set()
        if ix < h.shape[0]:
            row = h[ix]; changes = np.r_[0,np.flatnonzero(row[1:] != row[:-1])+1,len(row)]
            spans = {(int(a),int(b),float(row[a])) for a,b in zip(changes[:-1],changes[1:]) if row[a] > -2.}
        for span in list(active):
            if span not in spans:
                start = active.pop(span); a,b,top = span
                result.append((start*scale,ix*scale,a*scale,b*scale,top,-2.))
        for span in sorted(spans):
            active.setdefault(span,ix)
    return result
