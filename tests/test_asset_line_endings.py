import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rob2nav.assets import source


class AssetChecksumTests(unittest.TestCase):
    def test_only_explicitly_declared_text_accepts_crlf(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);root=base/'Rob2Nav';root.mkdir()
            cache=base/'Rob2Nav-upstreams/cache';cache.mkdir(parents=True)
            file=cache/'source.py';file.write_bytes(b'a=1\r\n')
            entry=dict(folder='cache',sha256={'source.py':hashlib.sha256(b'a=1\n').hexdigest()},lf_normalized_files=['source.py'])
            lock=base/'lock.json';lock.write_text(json.dumps({'pie':entry}))
            with patch('rob2nav.assets.LOCK',lock):
                self.assertEqual(source(root,'pie')[0],cache)
                file.write_bytes(b'a=2\r\n')
                with self.assertRaises(ValueError):source(root,'pie')


if __name__=='__main__':unittest.main()
