"""Warm local Sherpa engines or SGLang-Omni HTTP; ASR and TTS are independent."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import urllib.request
import uuid
import wave


def wav_info(data, max_seconds=15):
    with wave.open(io.BytesIO(data), 'rb') as stream:
        rate, frames = stream.getframerate(), stream.getnframes()
        if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getcomptype() != 'NONE':
            raise ValueError('需要单声道 PCM16 WAV')
        if not 8000 <= rate <= 48000 or not .1 <= frames/rate <= max_seconds:
            raise ValueError('语音需为 0.1–15 秒、8–48 kHz')
        raw = stream.readframes(frames)
        if len(raw) != frames*2: raise ValueError('WAV 数据被截断')
        return rate, frames/rate, raw


def encode_wav(samples, rate):
    import numpy as np
    raw = (np.clip(samples, -1, 1)*32767).astype('<i2').tobytes()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(rate); stream.writeframes(raw)
    return buf.getvalue()


class Speech:
    def __init__(self, config=None, root=None):
        self.config = config or {}; self.root = Path(root or '.')
        self.asr = None; self.tts = None
        self.asr_lock = threading.Lock(); self.tts_lock = threading.Lock()
        self.cache = collections.OrderedDict(); self.cache_limit = 64
        a, t = self.config.get('asr', {}), self.config.get('tts', {})
        if a.get('backend') == 'sherpa':
            import sherpa_onnx
            folder = self.root/a['directory']
            self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(folder/'model.int8.onnx'), tokens=str(folder/'tokens.txt'),
                num_threads=a.get('threads', 2), provider=a.get('provider', 'cpu'),
                language=a.get('language', 'zh'), use_itn=True, debug=False)
        if t.get('backend') == 'sherpa':
            import sherpa_onnx
            folder = self.root/t['directory']
            rules = ','.join(str(folder/name) for name in ('date.fst', 'number.fst') if (folder/name).exists())
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=str(folder/'model.onnx'),
                        lexicon=str(folder/'lexicon.txt'), tokens=str(folder/'tokens.txt'),
                        dict_dir=str(folder/'dict') if (folder/'dict').is_dir() else '',
                        noise_scale=t.get('noise_scale',0.0),noise_scale_w=t.get('noise_scale_w',0.0)),
                    num_threads=t.get('threads', 2), provider=t.get('provider', 'cpu'), debug=False),
                rule_fsts=rules, max_num_sentences=1)
            if not cfg.validate(): raise ValueError('TTS 模型配置校验失败')
            self.tts = sherpa_onnx.OfflineTts(cfg)

    def info(self):
        return dict(asr=self.config.get('asr', {}).get('backend', 'none'),
                    tts=self.config.get('tts', {}).get('backend', 'none'),
                    tts_cache_entries=len(self.cache), input='push-to-talk PCM16 WAV; final utterances only')

    def _http(self, config, route, data, content_type):
        base = config['base_url'].rstrip('/')
        if not base.startswith(('http://', 'https://')): raise ValueError('语音服务 URL 必须为 HTTP(S)')
        headers = {'Content-Type': content_type}
        key = os.environ.get(config.get('api_key_env', 'SGLANG_API_KEY'), '')
        if key: headers['Authorization'] = 'Bearer '+key
        req = urllib.request.Request(base+route, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=config.get('timeout_s', 15)) as response:
            result = response.read(16*1024*1024+1)
        if len(result) > 16*1024*1024: raise ValueError('语音响应过大')
        return result

    def transcribe(self, data):
        rate, duration, raw = wav_info(data)
        if not self.asr_lock.acquire(blocking=False): raise RuntimeError('正在识别上一条语音，请稍后重试')
        started = time.perf_counter()
        try:
            cfg = self.config.get('asr', {}); kind = cfg.get('backend')
            if kind == 'sherpa':
                import numpy as np
                samples = np.frombuffer(raw, dtype='<i2').astype('float32')/32768
                stream = self.asr.create_stream(); stream.accept_waveform(rate, samples)
                self.asr.decode_stream(stream); text = stream.result.text
            elif kind == 'sglang':
                boundary = 'voice-'+uuid.uuid4().hex
                fields = dict(model=cfg['model'], response_format='json', language=cfg.get('language', 'zh'))
                body = b''
                for key, value in fields.items():
                    body += ('--'+boundary+'\r\nContent-Disposition: form-data; name="'+key+'"\r\n\r\n'+value+'\r\n').encode()
                body += ('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="utterance.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+data
                body += ('\r\n--'+boundary+'--\r\n').encode()
                text = json.loads(self._http(cfg, '/audio/transcriptions', body, 'multipart/form-data; boundary='+boundary))['text']
            else: raise RuntimeError('尚未配置 ASR；下载模型并选择 speech 配置，文本控制仍可使用')
            if not isinstance(text, str) or len(text) > 1000: raise ValueError('非法识别结果')
            return dict(text=text, asr_ms=(time.perf_counter()-started)*1000, audio_s=duration)
        finally: self.asr_lock.release()

    def synthesize(self, text):
        if not isinstance(text, str) or not 0 < len(text) <= 120: raise ValueError('回复语音限 1–120 字符')
        started = time.perf_counter()
        with self.tts_lock:
            if text in self.cache:
                data = self.cache.pop(text); self.cache[text] = data
                return data, dict(tts_ms=(time.perf_counter()-started)*1000, cache_hit=True)
            cfg = self.config.get('tts', {}); kind = cfg.get('backend')
            if kind == 'sherpa':
                import sherpa_onnx
                # GenerationConfig is used by current API; legacy generate is supported as fallback.
                if hasattr(sherpa_onnx, 'GenerationConfig'):
                    gen = sherpa_onnx.GenerationConfig(); gen.sid = 0; gen.speed = cfg.get('speed', 1.0)
                    audio = self.tts.generate(text, gen)
                else: audio = self.tts.generate(text, sid=0, speed=cfg.get('speed', 1.0))
                if not len(audio.samples): raise RuntimeError('TTS 返回空音频')
                data = encode_wav(audio.samples, audio.sample_rate)
            elif kind == 'sglang':
                body = json.dumps(dict(model=cfg['model'], input=text, voice=cfg.get('voice', 'default'),
                                       response_format='wav', stream=False), ensure_ascii=False).encode()
                data = self._http(cfg, '/audio/speech', body, 'application/json')
                with wave.open(io.BytesIO(data)) as check:
                    if check.getnframes() == 0: raise ValueError('TTS 返回空 WAV')
            else: raise RuntimeError('尚未配置 TTS，可在页面阅读回复')
            self.cache[text] = data
            while len(self.cache) > self.cache_limit: self.cache.popitem(last=False)
            return data, dict(tts_ms=(time.perf_counter()-started)*1000, cache_hit=False)

    def prewarm(self, replies=None):
        """Warm recognition and cache common short replies before accepting control."""
        if self.config.get('asr', {}).get('backend') != 'none' and self.config.get('asr'):
            self.transcribe(encode_wav([0.]*16000, 16000))
        if self.config.get('tts', {}).get('backend') != 'none' and self.config.get('tts'):
            fixed=['控制已启用，可以说地点或技能。', '正在停止并等待稳定站立。',
                   '已停止并稳定站立。', '动作结束，正在恢复站立。', '技能已完成并恢复站立。']
            for text in fixed+list(replies or [])[:self.cache_limit-len(fixed)]:
                self.synthesize(text)
