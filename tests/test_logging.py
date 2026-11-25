"""Logging behavior for library users and command-line runs."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from ffthompy import PrintControl, Timer


def test_timer_logs_elapsed_time(caplog, capsys):
    timer = Timer("logging test")
    with caplog.at_level(logging.INFO, logger="ffthompy"):
        timer.measure()

    assert len(timer.vals) == 1
    assert all(value >= 0 for value in timer.vals[0])
    assert caplog.record_tuples == [("ffthompy.general.base", logging.INFO, str(timer))]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("options", [{"log_time": False}, {"print_time": False}])
def test_timer_can_record_without_logging(options, caplog):
    timer = Timer()
    with caplog.at_level(logging.INFO, logger="ffthompy"):
        timer.measure(**options)

    assert len(timer.vals) == 1
    assert not caplog.records


def test_print_control_mutes_library_logs_and_restores_level(caplog):
    control = PrintControl()
    original_stdout = sys.stdout
    with caplog.at_level(logging.INFO, logger="ffthompy"):
        try:
            control.disable()
            control.disable()
            Timer("muted").measure()
            logging.getLogger("other_application").warning("Still visible")
            assert sys.stdout is original_stdout
        finally:
            control.enable()
        control.enable()
        Timer("restored").measure()
        assert logging.getLogger("ffthompy").level == logging.INFO

    assert [record.name for record in caplog.records] == [
        "other_application",
        "ffthompy.general.base",
    ]
    assert "restored" in caplog.records[-1].getMessage()
    assert sys.stdout is original_stdout


def test_inactive_print_control_keeps_logs_enabled(caplog):
    control = PrintControl(flag=False)
    with caplog.at_level(logging.INFO, logger="ffthompy"):
        try:
            control.disable()
            Timer("visible").measure()
        finally:
            control.enable()

    assert len(caplog.records) == 1
    assert "visible" in caplog.records[0].getMessage()


def test_import_preserves_application_logging_configuration():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
root = logging.getLogger()
handler = logging.NullHandler()
root.addHandler(handler)
root.setLevel(logging.ERROR)
import ffthompy.problem
assert root.handlers == [handler]
assert root.level == logging.ERROR
""",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
    assert result.stderr == ""


def test_command_line_logs_results():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "main.py"), "examples/scalar/from_file.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == ""
    assert "Homogenized matrices" in result.stderr
    assert "The calculation is finished!" in result.stderr
