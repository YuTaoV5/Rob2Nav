"""Terminal key bindings shared by Go2 Edu teleop and offline tests."""

VELOCITY_KEYS={
    'w':(.18,0.,0.), 's':(-.12,0.,0.),
    'a':(0.,.10,0.), 'd':(0.,-.10,0.),
    'q':(0.,0.,.35), 'e':(0.,0.,-.35),
    ' ':(0.,0.,0.),
}
SKILL_KEYS={'1':'stand','2':'walk_moe','3':'sideflip69','4':'backflip60','5':'upright52'}


def key_request(key):
    key=key.lower()
    if key in SKILL_KEYS:return {'switch':SKILL_KEYS[key]}
    if key=='t':return {'trigger':True}
    if key=='\x1b':return {'stop':True}
    if key in VELOCITY_KEYS:return {'velocity':VELOCITY_KEYS[key]}
    return None
