from __future__ import annotations

import os
import types
import unittest
from pathlib import Path
from unittest import mock

import verilume.cli as cli
from verilume.cli import (
    STREAMLIT_BROWSER_ADDRESS,
    STREAMLIT_HOST,
    STREAMLIT_PORT,
    STREAMLIT_PORT_ENV,
    _first_available_port,
    _disable_streamlit_development_mode,
    _resolve_streamlit_port,
    _streamlit_cli_args,
)


class StreamlitLaunchTests(unittest.TestCase):
    def test_streamlit_args_pin_localhost_port(self) -> None:
        args = _streamlit_cli_args(Path("src/verilume/app.py"))

        self.assertEqual(args[:2], ["run", "src/verilume/app.py"])
        self.assertEqual(args[args.index("--server.address") + 1], STREAMLIT_HOST)
        self.assertEqual(args[args.index("--server.port") + 1], str(STREAMLIT_PORT))
        self.assertEqual(
            args[args.index("--browser.serverAddress") + 1],
            STREAMLIT_BROWSER_ADDRESS,
        )
        self.assertEqual(args[args.index("--server.fileWatcherType") + 1], "none")
        self.assertEqual(args[args.index("--browser.gatherUsageStats") + 1], "false")

    def test_streamlit_args_accept_selected_port(self) -> None:
        args = _streamlit_cli_args(Path("src/verilume/app.py"), port=8512)

        self.assertEqual(args[args.index("--server.port") + 1], "8512")

    def test_first_available_port_skips_busy_ports(self) -> None:
        with mock.patch("verilume.cli._port_is_available", side_effect=[False, True]):
            self.assertEqual(_first_available_port(STREAMLIT_HOST, STREAMLIT_PORT), STREAMLIT_PORT + 1)

    def test_resolve_streamlit_port_uses_environment_override(self) -> None:
        with mock.patch.dict("os.environ", {STREAMLIT_PORT_ENV: "8525"}):
            self.assertEqual(_resolve_streamlit_port(), 8525)

    def test_resolve_streamlit_port_reclaims_default_when_preferred_ports_are_busy(self) -> None:
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("verilume.cli._first_available_port", return_value=None),
            mock.patch("verilume.cli._force_reclaim_streamlit_port", return_value=STREAMLIT_PORT) as reclaim,
        ):
            self.assertEqual(_resolve_streamlit_port(), STREAMLIT_PORT)
            reclaim.assert_called_once_with(STREAMLIT_HOST, STREAMLIT_PORT)

    def test_frozen_run_sets_development_mode_before_streamlit_patch(self) -> None:
        streamlit_module = types.ModuleType("streamlit")
        streamlit_web_module = types.ModuleType("streamlit.web")
        streamlit_cli_module = types.ModuleType("streamlit.web.cli")
        streamlit_cli_module.main = lambda: 0
        streamlit_module.web = streamlit_web_module
        streamlit_web_module.cli = streamlit_cli_module

        def assert_frozen_config_ready() -> None:
            self.assertEqual(os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"], "false")
            self.assertIn("--global.developmentMode", cli.sys.argv)
            index = cli.sys.argv.index("--global.developmentMode")
            self.assertEqual(cli.sys.argv[index + 1], "false")

        with (
            mock.patch.object(cli.sys, "frozen", True, create=True),
            mock.patch("verilume.cli._resolve_streamlit_port", return_value=8515),
            mock.patch(
                "verilume.cli._patch_streamlit_for_frozen_bundle",
                side_effect=assert_frozen_config_ready,
            ),
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch.dict(
                "sys.modules",
                {
                    "streamlit": streamlit_module,
                    "streamlit.web": streamlit_web_module,
                    "streamlit.web.cli": streamlit_cli_module,
                },
            ),
        ):
            self.assertEqual(cli.run_streamlit(), 0)

    def test_disable_streamlit_development_mode_updates_template_option(self) -> None:
        class FakeOption:
            def __init__(self) -> None:
                self.calls: list[tuple[bool, str]] = []

            def set_value(self, value: bool, where_defined: str) -> None:
                self.calls.append((value, where_defined))

        option = FakeOption()
        config = types.SimpleNamespace(
            _global_development_mode=option,
            _config_options=None,
        )
        development = types.SimpleNamespace(is_development_mode=True)

        _disable_streamlit_development_mode(config, development)

        self.assertEqual(option.calls, [(False, "<streamlit>")])
        self.assertFalse(development.is_development_mode)


if __name__ == "__main__":
    unittest.main()
