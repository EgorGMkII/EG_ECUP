from scripts import datasphere_runner


def test_launch_job_uses_synchronous_streaming(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_stream(command: list[str], _token: str) -> str:
        captured["command"] = command
        return "created job `bt1abcdefghijklmnop`\nSUCCESS\n"

    monkeypatch.setattr(datasphere_runner, "stream_command", fake_stream)
    job_id = datasphere_runner.launch_job("datasphere.test.yaml", "project-id", "token")

    assert job_id == "bt1abcdefghijklmnop"
    assert "--async" not in captured["command"]
    assert captured["command"][-4:] == ["-p", "project-id", "-c", "datasphere.test.yaml"]


def test_job_id_parser_ignores_project_and_operation_ids() -> None:
    output = "project bt1project operation bt1operation created job `bt1actualjob`"
    assert datasphere_runner.extract_job_id(output) == "bt1actualjob"


def test_resolve_pre_run_sha_honors_explicit_value() -> None:
    assert datasphere_runner.resolve_pre_run_sha("abc123") == "abc123"
