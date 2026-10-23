"""Tests of scripts/verify_pypi_release.py (the release check before the notification of
SlicerPictologics) and of the order of the release workflows. No test sends a request: a
fake opener gives the PyPI answers, and a fake clock counts the waits."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("verify_pypi_release", ROOT / "scripts" / "verify_pypi_release.py")  # fmt: skip
assert _SPEC is not None and _SPEC.loader is not None
release = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = release  # the dataclass reads its module
_SPEC.loader.exec_module(release)

VERSION = "0.8.0"
FILENAME = "pictologics-0.8.0-py3-none-any.whl"
SHA = "a" * 64
WHEEL = release.Wheel(VERSION, FILENAME, SHA)


def _document(files: Optional[list[dict[str, Any]]] = None, **info: Any) -> dict[str, Any]:
    """The PyPI JSON of release 0.8.0 with the wheel and the sdist, or with `files`."""
    if files is None:
        files = [
            {
                "filename": FILENAME,
                "packagetype": "bdist_wheel",
                "digests": {"sha256": SHA},
                "yanked": False,
            },  # fmt: skip
            {
                "filename": "pictologics-0.8.0.tar.gz",
                "packagetype": "sdist",
                "digests": {"sha256": "b" * 64},
                "yanked": False,
            },  # fmt: skip
        ]
    return {"info": {"name": "pictologics", "version": VERSION, **info}, "urls": files}


class _Response(io.BytesIO):
    """A response body that works as a context manager, as urlopen gives it."""


def _http_error(code: int, retry_after: Optional[str] = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://pypi.org", code, "error", headers, None)


class _PyPI:
    """A fake opener and clock: each request takes the next answer (a document, raw bytes
    or an exception) and one second; each wait moves the clock."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.now = 100.0
        self.waits: list[float] = []
        self.requests: list[tuple[str, float, str]] = []
        self.logged: list[str] = []

    def opener(self, request: Any, timeout: float) -> _Response:
        self.requests.append((request.full_url, timeout, request.get_header("User-agent")))
        self.now += 1.0
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        return _Response(answer if isinstance(answer, bytes) else json.dumps(answer).encode())

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def options(self) -> dict[str, Callable[..., Any]]:
        return {"opener": self.opener, "clock": self.clock, "sleep": self.sleep, "log": self.logged.append}  # fmt: skip


def test_a_matching_wheel_is_ready_at_the_first_try(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    pypi = _PyPI(_document())
    output = tmp_path / "github_output"
    code = release.main(["confirm", "--version", VERSION, "--sha256", SHA, "--github-output", str(output)], **pypi.options())  # fmt: skip
    assert code == 0 and pypi.waits == []
    url, timeout, agent = pypi.requests[0]
    assert url == "https://pypi.org/pypi/pictologics/0.8.0/json"
    assert 0 < timeout <= release.REQUEST_TIMEOUT_SECONDS and agent.startswith("pictologics-release-check/")  # fmt: skip
    assert output.read_text() == f"version={VERSION}\nfilename={FILENAME}\nsha256={SHA}\n"
    assert capsys.readouterr().out == f"ready: pictologics {VERSION} {FILENAME} sha256={SHA}\n"


def test_an_absent_release_gets_bounded_tries_until_the_wheel_appears() -> None:
    # HTTP 404, then only the sdist, then the wheel: two waits of 5 s and 10 s
    sdist_only = _document(files=_document()["urls"][1:])
    pypi = _PyPI(_http_error(404), sdist_only, _document())
    release.confirm(WHEEL, **pypi.options())
    assert pypi.waits == [5.0, 10.0] and len(pypi.requests) == 3
    assert "HTTP 404" in pypi.logged[0] and "does not show" in pypi.logged[1]


@pytest.mark.parametrize(
    "answer",
    [_http_error(404), _document(files=_document()["urls"][1:]), _document(files=[])],
)
def test_a_wheel_that_stays_absent_fails_at_the_deadline(tmp_path: Path, answer: Any) -> None:
    pypi = _PyPI(answer)
    output = tmp_path / "github_output"
    code = release.main(["confirm", "--version", VERSION, "--sha256", SHA, "--github-output", str(output)], **pypi.options())  # fmt: skip
    assert code == 1 and not output.exists()  # no readiness output
    assert pypi.waits == [5.0, 10.0, 20.0, 30.0, 30.0, 30.0, 30.0]  # capped at 30 s
    assert pypi.now - 100.0 <= release.DEADLINE_SECONDS + release.REQUEST_TIMEOUT_SECONDS


def test_transient_errors_get_bounded_tries_and_respect_retry_after() -> None:
    # A timeout, a failed connection, HTTP 503, HTTP 429 with Retry-After 2 and with 120
    # (capped at 30 s), and an HTTP date in Retry-After (the normal wait)
    pypi = _PyPI(
        TimeoutError("timed out"),
        urllib.error.URLError("connection refused"),
        _http_error(503),
        _http_error(429, "2"),
        _http_error(429, "120"),
        _http_error(502, "Wed, 21 Oct 2026 07:28:00 GMT"),
        _document(),
    )
    release.confirm(WHEEL, deadline=900.0, **pypi.options())
    assert pypi.waits == [5.0, 10.0, 20.0, 2.0, 30.0, 30.0]
    assert any("HTTP 429" in line for line in pypi.logged)


def test_a_short_deadline_bounds_the_request_timeout() -> None:
    pypi = _PyPI(_http_error(500))
    with pytest.raises(release.ReleaseError, match="after 3 s PyPI still does not show"):
        release.confirm(WHEEL, deadline=3.0, **pypi.options())
    assert pypi.waits == [] and pypi.requests[0][1] == 3.0


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (_document(files=[{**_document()["urls"][0], "yanked": True}]), "yanked"),
        (_document(files=[{**_document()["urls"][0], "yanked": None}]), "yanked"),
        (_document(version="0.8.1"), "not pictologics 0.8.0"),
        (_document(name="other"), "not pictologics 0.8.0"),
        (_document(name=None), "not pictologics 0.8.0"),
        (
            _document(files=[{**_document()["urls"][0], "digests": {"sha256": "c" * 64}}]),
            "not the built",
        ),  # fmt: skip
        (_document(files=[{**_document()["urls"][0], "digests": None}]), "not the built"),
        (_document(files=[{**_document()["urls"][0], "packagetype": "sdist"}]), "the type 'sdist'"),
        (_document(files=_document()["urls"][:1] * 2), "2 times"),
        (b"<html>not json</html>", "no valid JSON"),
        ({"info": "x", "urls": []}, "not the JSON of a release"),
        ([1, 2], "not the JSON of a release"),
        (_http_error(403), "HTTP 403"),
    ],
)
def test_a_conflict_fails_at_once(answer: Any, message: str) -> None:
    pypi = _PyPI(answer)
    with pytest.raises(release.ReleaseError, match=message):
        release.confirm(WHEEL, **pypi.options())
    assert pypi.waits == [] and len(pypi.requests) == 1


@pytest.mark.parametrize("version", ["0.8.0rc1", "0.8.0-rc1", "v0.8.0", "0.8", "08.0.0", "0.8.0.post1", " 0.8.0", ""])  # fmt: skip
def test_a_version_that_is_not_stable_is_rejected_before_any_request(version: str) -> None:
    pypi = _PyPI(_document())
    assert release.main(["confirm", "--version", version, "--sha256", SHA], **pypi.options()) == 1  # fmt: skip
    assert pypi.requests == []


def test_a_bad_digest_or_file_name_is_rejected_before_any_request() -> None:
    pypi = _PyPI(_document())
    for extra in (["--sha256", "A" * 64], ["--sha256", SHA[:-1]], ["--sha256", SHA, "--filename", "pictologics-0.8.0.tar.gz"]):  # fmt: skip
        assert release.main(["confirm", "--version", VERSION, *extra], **pypi.options()) == 1
    assert pypi.requests == []
    assert release.trusted_wheel(VERSION, FILENAME, SHA) == WHEEL


def test_the_deadline_option_is_bounded() -> None:
    for value in ("0", "901", "nan"):
        with pytest.raises(SystemExit):
            release.main(["confirm", "--version", VERSION, "--sha256", SHA, "--deadline", value])
    pypi = _PyPI(_http_error(404))
    code = release.main(["confirm", "--version", VERSION, "--sha256", SHA, "--deadline", "12"], **pypi.options())  # fmt: skip
    assert code == 1 and pypi.waits == [5.0]  # the next wait of 10 s passes the deadline


def test_the_script_runs_alone_as_the_workflow_runs_it(tmp_path: Path) -> None:
    # A new interpreter in isolated mode, with no Pictologics import: the record command
    path = _wheel(tmp_path)
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts" / "verify_pypi_release.py"),
            "record",
            "--version",
            VERSION,
            "--wheel",
            str(path),
        ],  # fmt: skip
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0 and done.stdout.startswith(f"recorded: pictologics {VERSION} {FILENAME}")  # fmt: skip


def _wheel(folder: Path, name: str = FILENAME, metadata: Optional[str] = None, count: int = 1) -> Path:  # fmt: skip
    path = folder / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pictologics/__init__.py", "")
        for index in range(count):
            text = metadata if metadata is not None else f"Metadata-Version: 2.4\nName: Pictologics\nVersion: {VERSION}\n"  # fmt: skip
            archive.writestr(f"pictologics-{index}.dist-info/METADATA", text)
    return path


def test_record_gives_the_name_and_the_sha256_of_the_built_wheel(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    path = _wheel(tmp_path)
    output = tmp_path / "github_output"
    assert release.main(["record", "--version", VERSION, "--wheel", str(path), "--github-output", str(output)]) == 0  # fmt: skip
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert output.read_text() == f"version={VERSION}\nfilename={FILENAME}\nsha256={digest}\n"
    assert (
        capsys.readouterr().out == f"recorded: pictologics {VERSION} {FILENAME} sha256={digest}\n"
    )


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (
            lambda d: _wheel(d, metadata="Name: pictologics\nVersion: 0.8.1\n"),
            "not pictologics 0.8.0",
        ),  # fmt: skip
        (
            lambda d: _wheel(d, metadata=f"Name: other\nVersion: {VERSION}\n"),
            "not pictologics 0.8.0",
        ),  # fmt: skip
        (lambda d: _wheel(d, count=0), "0 METADATA files"),
        (lambda d: _wheel(d, count=2), "2 METADATA files"),
        (lambda d: _wheel(d, name="pictologics-0.8.1-py3-none-any.whl"), "not pictologics-0.8.0"),
        (lambda d: d / FILENAME, "cannot read the wheel"),  # absent
        (
            lambda d: (d / FILENAME).write_bytes(b"not a zip") and d / FILENAME,
            "cannot read the wheel",
        ),  # fmt: skip
    ],
)
def test_record_rejects_a_wheel_that_does_not_match(tmp_path: Path, make: Callable[[Path], Path], message: str) -> None:  # fmt: skip
    output = tmp_path / "github_output"
    path = make(tmp_path)
    with pytest.raises(release.ReleaseError, match=message):
        release.record(VERSION, path)
    assert release.main(["record", "--version", VERSION, "--wheel", str(path), "--github-output", str(output)]) == 1  # fmt: skip
    assert not output.exists()


def _workflow(name: str) -> dict[Any, Any]:
    document = yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text())
    assert isinstance(document, dict)
    return document


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads the key "on" as True
    assert isinstance(triggers, dict)
    return triggers


def test_the_notification_waits_for_the_upload_and_has_one_automatic_path() -> None:
    publish = _workflow("publish-pypi.yml")
    assert _triggers(publish) == {"release": {"types": ["released"]}}
    upload_job = publish["jobs"]["publish"]
    assert upload_job["environment"] == "pypi" and upload_job["permissions"]["id-token"] == "write"
    assert upload_job["if"] == "${{ !github.event.release.prerelease }}"
    steps = [step.get("name") for step in upload_job["steps"]]
    order = ["Run tests", "Check the version", "Build package", "Record the wheel", "Publish to PyPI (Trusted Publisher)"]  # fmt: skip
    assert [steps.index(name) for name in order] == sorted(steps.index(name) for name in order)
    upload = upload_job["steps"][steps.index("Publish to PyPI (Trusted Publisher)")]
    assert upload["uses"].startswith("pypa/gh-action-pypi-publish@") and "with" not in upload
    # The notification job needs the upload job, with no condition that would run it after a
    # failure, and gets the record of the built wheel
    notify_job = publish["jobs"]["notify"]
    assert notify_job["needs"] == "publish" and "if" not in notify_job
    assert notify_job["uses"] == "./.github/workflows/notify-slicer-extension.yml"
    assert set(notify_job["with"]) == {"version", "filename", "sha256"}
    assert all(f"needs.publish.outputs.{key}" in value for key, value in notify_job["with"].items())  # fmt: skip
    # The notification workflow has no automatic trigger of its own
    notify = _workflow("notify-slicer-extension.yml")
    assert set(_triggers(notify)) == {"workflow_call", "workflow_dispatch"}
    jobs = notify["jobs"]
    assert jobs["dispatch"]["needs"] == "confirm" and "if" not in jobs["dispatch"]
    assert "SLICER_EXTENSION_DISPATCH_TOKEN" in jobs["dispatch"]["env"]["DISPATCH_TOKEN"]
    assert "SLICER_EXTENSION_DISPATCH_TOKEN" not in json.dumps(jobs["confirm"])
    confirm_run = next(step["run"] for step in jobs["confirm"]["steps"] if step.get("id") == "pypi")
    assert "args=(confirm " in confirm_run and "scripts/verify_pypi_release.py" in confirm_run
    assert "${{" not in confirm_run  # the inputs come in through the environment
    # No other workflow notifies the extension, and TestPyPI does not
    for path in (ROOT / ".github" / "workflows").glob("*.yml"):
        if path.name not in ("publish-pypi.yml", "notify-slicer-extension.yml"):
            text = path.read_text()
            assert "notify-slicer-extension" not in text and "repository-dispatch" not in text, path.name  # fmt: skip
