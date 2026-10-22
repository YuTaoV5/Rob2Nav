import io
import urllib.error
from unittest.mock import patch
import pytest
from studio.pipeline_api import request


def test_get_recovers_from_service_restart():
    with patch('urllib.request.urlopen',side_effect=[urllib.error.URLError('refused'),io.BytesIO(b'[]')]) as opened,patch('time.sleep') as sleep:
        assert request('jobs')==[]
        assert opened.call_count==2
        sleep.assert_called_once_with(1)


def test_unknown_job_creation_outcome_is_never_retried():
    with patch('urllib.request.urlopen',side_effect=urllib.error.URLError('refused')) as opened,patch('time.sleep') as sleep:
        with pytest.raises(urllib.error.URLError):request('jobs',{'name':'same-job'})
        assert opened.call_count==1
        sleep.assert_not_called()


def test_validation_error_is_not_hidden_by_retry():
    e=urllib.error.HTTPError('http://test',422,'invalid',{},None)
    with patch('urllib.request.urlopen',side_effect=e) as opened,patch('time.sleep') as sleep:
        with pytest.raises(urllib.error.HTTPError):request('preflight',{})
        assert opened.call_count==1
        sleep.assert_not_called()
