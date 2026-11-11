"""Keep robot self contacts enabled apart from two overlapping near-joint mesh pairs."""

NOMINAL_OVERLAPS = {("FL_thigh", "base_link"), ("FR_thigh", "base_link")}


def exclude_nominal_overlaps(spec):
    for first, second in sorted(NOMINAL_OVERLAPS):
        spec.add_exclude(bodyname1=first, bodyname2=second)
    return spec
