"""Position-target correction near the existing AS2-W hard angle guard."""


def guarded_target(target, position, velocity, lower, upper, *, margin=.05):
    # Anticipate motion over two policy periods and move the PD target inward.
    predicted = position+.040*velocity
    lower_risk = (lower+.10-predicted).clip(0.,.12)
    upper_risk = (predicted-(upper-.10)).clip(0.,.12)
    return (target+2.*(lower_risk-upper_risk)).clip(lower+margin,upper-margin)
