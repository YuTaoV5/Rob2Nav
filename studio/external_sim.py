"""Compatibility entry point; deployment backends live in rob2nav."""
from rob2nav.backends.source_mujoco import pie, lsd, main

if __name__=="__main__":main()
