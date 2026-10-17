"""Tests for deployment server configuration."""

import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maintsense.server import configured_port, main  # noqa: E402


class ServerConfigurationTests(unittest.TestCase):
    def test_uses_platform_port_and_default(self):
        self.assertEqual(configured_port("8123"), 8123)
        self.assertEqual(configured_port(None), 8000)

    def test_rejects_invalid_or_out_of_range_port(self):
        for value in ("not-a-port", "0", "65536"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                configured_port(value)

    @patch("maintsense.server.uvicorn.run")
    @patch.dict("os.environ", {"PORT": "8124"})
    def test_production_entrypoint_uses_port_without_reload(self, run_server):
        main()

        run_server.assert_called_once_with(
            "maintsense.api:app",
            host="0.0.0.0",
            port=8124,
            reload=False,
        )


if __name__ == "__main__":
    unittest.main()
