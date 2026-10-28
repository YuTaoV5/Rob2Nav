#!/usr/bin/env python3
"""Measure actual ASR/TTS latency. No 5090 numbers are fabricated."""
import argparse
import json
import platform
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_agent.speech import Speech


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--wav',type=Path,help='Optional actual PTT sample, mono PCM16')
    p.add_argument('--text',default='正在前往会议室。');p.add_argument('--runs',type=int,default=5)
    p.add_argument('--output',type=Path,default=Path('data/voice/latency.json'));args=p.parse_args()
    if not 2<=args.runs<=100:p.error('runs must be 2–100')
    root=Path(__file__).resolve().parents[1];speech=Speech(json.loads(args.config.read_text(encoding='utf-8')),root)
    measurements=[]
    for i in range(args.runs):
        row={'run':i,'phase':'cold' if i==0 else 'warm'}
        if args.wav:row['asr']=speech.transcribe(args.wav.read_bytes())
        # Measure warmed model generation separately from exact-reply cache hits.
        speech.cache.pop(args.text,None)
        _,row['tts_uncached']=speech.synthesize(args.text)
        _,row['tts_cached']=speech.synthesize(args.text);measurements.append(row)
    result=dict(platform=platform.platform(),config=json.loads(args.config.read_text(encoding='utf-8')),measurements=measurements,
        note='ASR timing excludes human speaking; TTS is full short-WAV generation/cache, not acoustic playback or streaming TTFA')
    if args.wav:result['warm_asr_median_ms']=statistics.median(x['asr']['asr_ms'] for x in measurements[1:])
    result['warm_tts_generation_median_ms']=statistics.median(x['tts_uncached']['tts_ms'] for x in measurements[1:])
    result['warm_tts_cache_median_ms']=statistics.median(x['tts_cached']['tts_ms'] for x in measurements[1:])
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
