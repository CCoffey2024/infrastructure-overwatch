import argparse
import sys
import types

from infrastructure_overwatch.cli import _cmd_serve


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "host": "127.0.0.1",
        "port": 8765,
        "workspace": "outputs",
        "weights": "outputs/weights/real_vehicle_detector.pt",
        "allow_remote": False,
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_serve_refuses_a_non_loopback_host_without_allow_remote(capsys):
    exit_code = _cmd_serve(_args(host="0.0.0.0"))
    assert exit_code == 1
    assert "allow-remote" in capsys.readouterr().err


def test_serve_accepts_loopback_hosts_without_allow_remote(tmp_path, monkeypatch, capsys):
    # _cmd_serve does `import uvicorn` locally (so the `web` extra stays optional for
    # everything else); stub sys.modules so that import resolves to a fake uvicorn.run
    # that doesn't actually bind a socket -- only the refusal gate is under test here.
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, host, port: None
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    for host in ("127.0.0.1", "localhost", "::1"):
        # a real (if unused) workspace dir under tmp_path -- must not touch the repo's own outputs/
        exit_code = _cmd_serve(_args(host=host, workspace=str(tmp_path / host.replace(":", "_"))))
        assert exit_code == 0
        assert "Refusing to bind" not in capsys.readouterr().err
