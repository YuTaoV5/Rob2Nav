"""Shared physical geometry for Isaac Lab training and MuJoCo comparison."""

def lane_boxes(difficulty, kind, size=(10., 3.)):
    if kind.startswith("ame_"):
        from .ame_terrain import boxes
        return boxes(difficulty, kind)
    length, width = size
    boxes = []

    def box(x0, x1, y0, y1, top, bottom=-.8):
        boxes.append((x0, x1, y0, y1, top, bottom))

    # Start at x=1.0; obstacle begins 1.6 m in front of the reset origin.
    start, end = 2.6, 6.2
    d = float(difficulty)
    if kind == "flat":
        box(0, length, 0, width, 0.)
    elif kind == "platform":
        h = .03 + .37*d
        box(0, start, 0, width, 0.)
        box(start, end, 0, width, h)
        box(end, length, 0, width, 0.)
    elif kind == "stairs":
        h, run = .025+.155*d, .40
        box(0, start, 0, width, 0.)
        for i in range(4):
            box(start+i*run, start+(i+1)*run, 0, width, (i+1)*h)
        box(start+4*run, end, 0, width, 4*h)
        for i in range(4):
            box(end+i*run, end+(i+1)*run, 0, width, (3-i)*h)
        box(end+4*run, length, 0, width, 0.)
    elif kind == "stones":
        # A real discontinuous support region: gaps extend down to -0.55 m.
        # Two staggered rows allow separate left/right footholds.
        box(0, length, 0, width, -.55)
        box(0, start, 0, width, 0.)
        box(end, length, 0, width, 0.)
        gap = .015+.145*d
        sx, sy = .58-.24*d, .52-.22*d
        pitch = sx+gap
        x = start
        index = 0
        while x < end:
            for side in (-1, 1):
                cy = width/2+side*(.20+.025*d)
                offset = (.06*d if side == 1 else 0.)
                top = (.015+.065*d)*((index+(side == 1)) % 3)/2
                box(x+offset, min(x+offset+sx, end+.1), cy-sy/2, cy+sy/2, top)
            x += pitch
            index += 1
    else:
        raise ValueError(kind)
    return boxes
