"""Record the wheel of a Pictologics release, and confirm that production PyPI shows it.

The release workflows use two commands. They need only the standard library: they do not
import Pictologics and compile no kernel.

record --version X.Y.Z --wheel PATH
    Check the version and the metadata of the built wheel, and give its file name and its
    SHA-256. The publish job runs it before the upload, so its output is the trusted record.

confirm --version X.Y.Z --sha256 HEX [--filename NAME] [--deadline SECONDS]
    Read https://pypi.org/pypi/pictologics/X.Y.Z/json until it shows that wheel with that
    SHA-256, not yanked. An absent release or wheel, a failed request, HTTP 429 and HTTP 5xx
    get more tries until the deadline. A conflict fails at once.

On success a command prints one record line and, with --github-output, adds the version,
the file name and the SHA-256 to that file. On failure it writes nothing there and exits
with 1.
"""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT = "pictologics"
PYPI_JSON = "https://pypi.org/pypi/{project}/{version}/json"
USER_AGENT = "pictologics-release-check/1 (+https://github.com/martonkolossvary/pictologics)"
# A stable version only: no pre-release, post-release, local or "v" part
STABLE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
SHA256 = re.compile(r"[0-9a-f]{64}")
DEADLINE_SECONDS = 180.0
MAX_DEADLINE_SECONDS = 900.0
REQUEST_TIMEOUT_SECONDS = 15.0
FIRST_WAIT_SECONDS = 5.0
MAX_WAIT_SECONDS = 30.0


class ReleaseError(Exception):
    """The release is not ready, and another try cannot change that."""


class NotYet(Exception):
    """PyPI does not show the wheel yet, or a request failed for a time. `wait` is the delay
    in seconds that the server asks for (Retry-After), or None."""

    def __init__(self, reason: str, wait: Optional[float] = None) -> None:
        super().__init__(reason)
        self.wait = wait


@dataclass(frozen=True)
class Wheel:
    """The trusted record of a built wheel."""

    version: str
    filename: str
    sha256: str


def _normalized(name: str) -> str:
    """A project name in the normalized form of PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def stable_version(text: str) -> str:
    """`text` when it is a stable version X.Y.Z, else a ReleaseError."""
    if not STABLE.fullmatch(text):
        raise ReleaseError(f"{text!r} is not a stable version X.Y.Z")
    return text


def wheel_filename(version: str) -> str:
    """The file name of the pure-Python wheel of `version`."""
    return f"{PROJECT}-{version}-py3-none-any.whl"


def record(version: str, path: Path) -> Wheel:
    """The record of the built wheel at `path`: its name, version and metadata must agree
    with `version`. The SHA-256 is computed from the file."""
    version = stable_version(version)
    if path.name != wheel_filename(version):
        raise ReleaseError(f"the wheel is {path.name}, not {wheel_filename(version)}")
    try:
        with zipfile.ZipFile(path) as archive:
            found = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(found) != 1:
                raise ReleaseError(f"the wheel {path.name} has {len(found)} METADATA files, not 1")
            text = archive.read(found[0]).decode("utf-8")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise ReleaseError(f"cannot read the wheel {path}: {error}") from error
    metadata = email.parser.Parser().parsestr(text)
    name, found_version = metadata.get("Name", ""), metadata.get("Version", "")
    if _normalized(name) != PROJECT or found_version != version:
        raise ReleaseError(
            f"the wheel metadata gives {name!r} {found_version!r}, not {PROJECT} {version}"
        )
    return Wheel(version, path.name, digest.hexdigest())


def trusted_wheel(version: str, filename: Optional[str], sha256: str) -> Wheel:
    """The record of a wheel from a version and a SHA-256 of a trusted source (the record
    line of the publish run). The file name must be the wheel name of the version."""
    version = stable_version(version)
    filename = wheel_filename(version) if filename is None else filename
    if filename != wheel_filename(version):
        raise ReleaseError(f"the file name is {filename}, not {wheel_filename(version)}")
    if not SHA256.fullmatch(sha256):
        raise ReleaseError(f"{sha256!r} is not a SHA-256 in 64 lowercase hex digits")
    return Wheel(version, filename, sha256)


def _retry_after(value: Optional[str]) -> Optional[float]:
    """The delay of a Retry-After header in seconds; None for no header or an HTTP date."""
    if value is None or not value.strip().isdigit():
        return None
    return float(value.strip())


def fetch(url: str, timeout: float, opener: Callable[..., Any]) -> Any:
    """The JSON document at `url`. An absent document (HTTP 404), HTTP 429, HTTP 5xx and a
    failed request raise NotYet; another HTTP error and a body that is not JSON raise
    ReleaseError."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:  # before URLError: HTTPError is a URLError
        if error.code == 404:
            raise NotYet("PyPI has no such release yet (HTTP 404)") from None
        if error.code == 429 or error.code >= 500:
            wait = _retry_after(error.headers.get("Retry-After") if error.headers else None)
            raise NotYet(f"PyPI answered HTTP {error.code}", wait) from None
        raise ReleaseError(f"PyPI answered HTTP {error.code}") from None
    except (urllib.error.URLError, OSError) as error:  # OSError holds TimeoutError
        raise NotYet(f"the request to PyPI failed: {error}") from None
    try:
        return json.loads(body)
    except ValueError as error:  # JSONDecodeError and UnicodeDecodeError
        raise ReleaseError(f"PyPI gave no valid JSON: {error}") from None


def check_release(document: Any, wheel: Wheel) -> None:
    """Return when the PyPI JSON of a release shows `wheel`: the same project, version, file
    name, package type and SHA-256, not yanked. Raise NotYet when the wheel is not there yet
    (for example only the sdist), and ReleaseError on any conflict."""
    if not (
        isinstance(document, dict)
        and isinstance(document.get("info"), dict)
        and isinstance(document.get("urls"), list)
    ):
        raise ReleaseError("the PyPI response is not the JSON of a release")
    info = document["info"]
    name, version = info.get("name"), info.get("version")
    if not isinstance(name, str) or _normalized(name) != PROJECT or version != wheel.version:
        raise ReleaseError(f"PyPI gives {name!r} {version!r}, not {PROJECT} {wheel.version}")
    files = [
        f for f in document["urls"] if isinstance(f, dict) and f.get("filename") == wheel.filename
    ]
    if not files:
        raise NotYet(f"PyPI does not show {wheel.filename} yet")
    if len(files) > 1:
        raise ReleaseError(f"PyPI shows {wheel.filename} {len(files)} times")
    entry = files[0]
    if entry.get("packagetype") != "bdist_wheel":
        raise ReleaseError(f"PyPI gives {wheel.filename} the type {entry.get('packagetype')!r}")
    digests = entry.get("digests")
    digest = digests.get("sha256") if isinstance(digests, dict) else None
    if digest != wheel.sha256:
        raise ReleaseError(
            f"PyPI gives {wheel.filename} the SHA-256 {digest!r}, not the built {wheel.sha256}"
        )
    if entry.get("yanked") is not False:
        raise ReleaseError(f"PyPI marks {wheel.filename} as yanked (or gives no yanked flag)")


def confirm(
    wheel: Wheel,
    deadline: float = DEADLINE_SECONDS,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> None:
    """Return when production PyPI shows `wheel` (see check_release). Another try follows a
    transient state after 5 s, then twice as long each time, at most 30 s or the delay that
    the server asks for, if smaller. No try starts after `deadline` seconds, and no request
    waits past it: then a ReleaseError says what PyPI showed last."""
    url = PYPI_JSON.format(project=PROJECT, version=wheel.version)
    end = clock() + deadline
    wait = FIRST_WAIT_SECONDS
    while True:
        timeout = max(1.0, min(REQUEST_TIMEOUT_SECONDS, end - clock()))
        try:
            check_release(fetch(url, timeout, opener), wheel)
            return
        except NotYet as state:
            pause = min(MAX_WAIT_SECONDS, wait if state.wait is None else state.wait)
            if clock() + pause >= end:
                raise ReleaseError(
                    f"{state}; after {deadline:.0f} s PyPI still does not show the wheel"
                ) from None
            log(f"not ready: {state}; next try in {pause:.0f} s")
            sleep(pause)
            wait = min(2.0 * wait, MAX_WAIT_SECONDS)


def _deadline(text: str) -> float:
    """A deadline in seconds from 1 to MAX_DEADLINE_SECONDS."""
    value = float(text)
    if not 1.0 <= value <= MAX_DEADLINE_SECONDS:
        raise argparse.ArgumentTypeError(
            f"the deadline must be from 1 to {MAX_DEADLINE_SECONDS:.0f} s"
        )
    return value


def main(argv: Optional[list[str]] = None, **confirm_options: Any) -> int:
    """Run a command (see the module docstring). `confirm_options` go to confirm."""
    parser = argparse.ArgumentParser(description="Record or confirm a Pictologics wheel on PyPI.")
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="record the built wheel")
    record_parser.add_argument("--version", required=True)
    record_parser.add_argument("--wheel", required=True, type=Path)
    confirm_parser = commands.add_parser("confirm", help="wait until PyPI shows the wheel")
    confirm_parser.add_argument("--version", required=True)
    confirm_parser.add_argument("--sha256", required=True)
    confirm_parser.add_argument("--filename")
    confirm_parser.add_argument("--deadline", type=_deadline, default=DEADLINE_SECONDS)
    for command in (record_parser, confirm_parser):
        command.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            wheel, label = record(args.version, args.wheel), "recorded"
        else:
            wheel = trusted_wheel(args.version, args.filename, args.sha256)
            confirm(wheel, args.deadline, **confirm_options)
            label = "ready"
    except ReleaseError as error:
        print(f"::error title=PyPI release check::{error}")
        return 1
    print(f"{label}: {PROJECT} {wheel.version} {wheel.filename} sha256={wheel.sha256}")
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(
                f"version={wheel.version}\nfilename={wheel.filename}\nsha256={wheel.sha256}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
