import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from voice_agent.speech import Speech,encode_wav,wav_info
from voice_agent.pcd import projection

ROOT=Path(__file__).resolve().parents[1]


class AudioPcdTests(unittest.TestCase):
    def test_pcm_roundtrip_and_invalid_channels(self):
        data=encode_wav([.1]*16000,16000);rate,duration,raw=wav_info(data)
        self.assertEqual(rate,16000);self.assertEqual(duration,1);self.assertEqual(len(raw),32000)
        with self.assertRaises(Exception):wav_info(b'not audio')
        buf=io.BytesIO()
        with wave.open(buf,'wb') as w:w.setnchannels(2);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0'*64000)
        with self.assertRaises(ValueError):wav_info(buf.getvalue())

    def test_long_audio_and_empty_tts_rejected(self):
        with self.assertRaises(ValueError):wav_info(encode_wav([0]*16000*16,16000))
        with self.assertRaises(ValueError):Speech().synthesize('')
        with self.assertRaises(RuntimeError):Speech().transcribe(encode_wav([0]*16000,16000))

    def test_tts_cache_avoids_repeat_backend(self):
        speech=Speech({'tts':{'backend':'sglang','base_url':'http://unused','model':'fake'}})
        calls=[]
        def request(*args):calls.append(args);return encode_wav([.1]*1600,16000)
        speech._http=request
        one,t1=speech.synthesize('已停止。');two,t2=speech.synthesize('已停止。')
        self.assertEqual(one,two);self.assertEqual(len(calls),1);self.assertFalse(t1['cache_hit']);self.assertTrue(t2['cache_hit'])

    def test_sglang_asr_multipart_contract(self):
        speech=Speech({'asr':{'backend':'sglang','base_url':'http://unused/v1','model':'model-id'}})
        def request(cfg,path,body,content_type):
            self.assertEqual(path,'/audio/transcriptions');self.assertIn(b'filename="utterance.wav"',body)
            self.assertIn(b'model-id',body);return '{"text":"去会议室"}'.encode()
        speech._http=request
        result=speech.transcribe(encode_wav([0]*16000,16000));self.assertEqual(result['text'],'去会议室')

    def test_binary_and_ascii_pcd_coordinates_and_slicing(self):
        folder=ROOT/'generated/test-tmp';folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            root=Path(temp);header='VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA '
            binary=root/'binary.pcd';binary.write_bytes((header+'binary\n').encode()+struct.pack('<ffffffff',1,2,.1,4,5,6,3.1,8))
            ascii_file=root/'ascii.pcd';ascii_file.write_text(header+'ascii\n1 2 0.1 4\n5 6 3.1 8\n')
            for path in (binary,ascii_file):
                result=projection(path,z_min=0,z_max=1);self.assertEqual(result['preview_points'],1)
                np.testing.assert_allclose(result['points'][0],[1,2,.1])
            binary.write_bytes((header+'binary\n').encode()+b'bad')
            with self.assertRaises(ValueError):projection(binary)


if __name__=='__main__':unittest.main()
