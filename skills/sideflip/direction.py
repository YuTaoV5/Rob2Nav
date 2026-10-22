"""Explicit sideflip direction; canonical unlabelled legacy references are left."""


def reference_direction(metadata):
    if not isinstance(metadata,dict):raise ValueError('Reference direction metadata must be an object')
    value=metadata.get('sideflip_direction')
    if value is None:return 'left'
    if value not in ('left','right'):raise ValueError('Sideflip direction must be left or right')
    return value


def require_reference_direction(metadata,requested):
    if requested not in ('left','right'):raise ValueError('Sideflip direction must be left or right')
    actual=reference_direction(metadata)
    if actual!=requested:raise ValueError(f'Sideflip reference direction is {actual}, requested {requested}')
    return requested


def inference_direction(reference_metadata,policy_metadata):
    requested=policy_metadata.get('sideflip_direction')
    if requested is None:requested=reference_direction(reference_metadata)
    return require_reference_direction(reference_metadata,requested)
