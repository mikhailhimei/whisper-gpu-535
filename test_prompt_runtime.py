import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from prompt_runtime import PromptStore, install_live_prompt, make_server


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'initial-prompt.json'
        self.store = PromptStore(self.path, 'old prompt')
        self.server = make_server(('127.0.0.1', 0), 'test-token', self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/initial-prompt'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def call(self, data=None, token='test-token'):
        req = Request(self.url, data=None if data is None else json.dumps(data).encode(),
                      method='GET' if data is None else 'PUT',
                      headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
        with urlopen(req, timeout=3) as response:
            return json.load(response)

    def test_update_is_live_for_existing_loader_and_survives_restart(self):
        class Loader:
            def __init__(self):
                self.initial_prompt = 'CLI prompt'
            def transcribe(self):
                return self.initial_prompt
        install_live_prompt(Loader, self.store)
        loader = Loader()
        self.assertEqual(loader.transcribe(), 'old prompt')
        result = self.call({'initial_prompt': 'Коридор, пылесос, коты.'})
        self.assertTrue(result['applied'])
        self.assertEqual(loader.transcribe(), 'Коридор, пылесос, коты.')
        self.assertEqual(self.call()['initial_prompt'], loader.transcribe())
        self.assertEqual(PromptStore(self.path, 'fallback').get(), loader.transcribe())

    def test_empty_prompt_disables_hint(self):
        self.call({'initial_prompt': ''})
        self.assertEqual(self.call()['initial_prompt'], '')
        self.assertEqual(PromptStore(self.path, 'fallback').get(), '')

    def test_wrong_token_does_not_change_prompt(self):
        with self.assertRaises(HTTPError) as result:
            self.call({'initial_prompt': 'bad'}, token='wrong')
        self.assertEqual(result.exception.code, 401)
        self.assertEqual(self.store.get(), 'old prompt')

    def test_invalid_payloads_do_not_change_prompt(self):
        for data in ({}, {'initial_prompt': 2}, {'initial_prompt': 'x' * 16001}, []):
            with self.subTest(data_type=type(data).__name__), self.assertRaises(HTTPError) as result:
                self.call(data)
            self.assertEqual(result.exception.code, 400)
        self.assertEqual(self.store.get(), 'old prompt')

    def test_failed_persistence_leaves_previous_value_active(self):
        self.store.set('saved')
        with patch('prompt_runtime.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(HTTPError) as result:
                self.call({'initial_prompt': 'unsaved'})
        self.assertEqual(result.exception.code, 500)
        self.assertEqual(self.store.get(), 'saved')
        self.assertEqual(PromptStore(self.path).get(), 'saved')
        self.assertFalse(list(self.path.parent.glob('.initial-prompt-*')))

    def test_empty_token_allows_get_and_put_without_authorization(self):
        server = make_server(('127.0.0.1', 0), '', self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}/initial-prompt'
        try:
            with urlopen(url, timeout=3) as response:
                self.assertEqual(json.load(response)['initial_prompt'], 'old prompt')
            request = Request(url, data=b'{"initial_prompt":"local"}', method='PUT',
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=3) as response:
                self.assertTrue(json.load(response)['applied'])
            self.assertEqual(PromptStore(self.path).get(), 'local')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_configured_token_rejects_missing_authorization(self):
        with self.assertRaises(HTTPError) as result:
            urlopen(self.url, timeout=3)
        self.assertEqual(result.exception.code, 401)


if __name__ == '__main__':
    unittest.main()
