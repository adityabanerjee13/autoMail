"""OAuth for a single mailbox.

Operational facts that are easy to lose and expensive to rediscover:

* The OAuth consent screen must be published ("In production"). While it is in
  "Testing", refresh tokens expire after 7 days and the daemon dies silently.
* gmail.readonly is a restricted scope. Expect an "unverified app"
  interstitial on first consent; click through. Full verification only matters
  if this is ever distributed to other users.
* Refresh tokens carrying Gmail scopes are invalidated when the Google account
  password changes. ``needs_reconnect()`` is what the UI's reconnect banner
  reads, and it will fire for exactly this reason one day.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from triage.config import Settings, get_settings

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


class ReconnectRequired(RuntimeError):
    """Stored credentials cannot be refreshed. A human must re-consent."""


@dataclass(slots=True)
class AuthStatus:
    connected: bool
    reason: str = ""
    scopes: list[str] | None = None


def _token_path(settings: Settings) -> Path:
    return settings.resolve(settings.gmail_token_path)


def _credentials_path(settings: Settings) -> Path:
    return settings.resolve(settings.gmail_credentials_path)


def load_credentials(settings: Settings | None = None) -> Credentials:
    """Load and refresh stored credentials, or raise ReconnectRequired."""
    settings = settings or get_settings()
    path = _token_path(settings)
    if not path.exists():
        raise ReconnectRequired(f"no stored token at {path}")

    creds = Credentials.from_authorized_user_file(str(path), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            raise ReconnectRequired(
                "refresh token rejected -- the account password may have changed, "
                "or the consent screen is still in Testing mode"
            ) from exc
        _write_token(path, creds)
        return creds
    raise ReconnectRequired("stored credentials are invalid and carry no refresh token")


def status(settings: Settings | None = None) -> AuthStatus:
    """Non-raising check for the health endpoint and the reconnect banner."""
    try:
        creds = load_credentials(settings)
    except ReconnectRequired as exc:
        return AuthStatus(connected=False, reason=str(exc))
    except Exception as exc:  # noqa: BLE001 - health must not raise
        return AuthStatus(connected=False, reason=repr(exc))
    return AuthStatus(connected=True, scopes=list(creds.scopes or []))


def run_consent_flow(settings: Settings | None = None, *, port: int = 0) -> Credentials:
    """Interactive consent. Run from a terminal on the machine, not the daemon.

    Desktop-app client type, so this opens a local loopback listener rather
    than needing a public redirect URI.
    """
    settings = settings or get_settings()
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds_path = _credentials_path(settings)
    if not creds_path.exists():
        raise FileNotFoundError(
            f"OAuth client secrets not found at {creds_path}. Create a Desktop app "
            "client in Google Cloud Console and download it there."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=port, prompt="consent", access_type="offline")
    _write_token(_token_path(settings), creds)
    return creds


# ---------------------------------------------------------------------------
# app password (IMAP)
# ---------------------------------------------------------------------------
#
# The second way in, and the one that needs no Google Cloud project at all:
# a 16-character App Password against Gmail's IMAP. It is what OpenClaw's
# imap-smtp-email skill uses, and it sidesteps the OAuth client creation that
# n8n also cannot avoid when self-hosted.
#
# The trade is real and worth stating plainly. An App Password is a standing
# credential with full mailbox access, it cannot be scoped to read-only the way
# gmail.readonly is, and it has to sit on disk in a form this process can read.
# What it does have is instant revocation: delete it in the Google account page
# and it is dead immediately, no token refresh to wait out. Prefer OAuth when
# you can be bothered with the console; use this when you cannot.


def _imap_path(settings: Settings) -> Path:
    return settings.resolve(settings.gmail_token_path).with_name("imap.json")


def normalize_imap_credentials(username: str, password: str) -> tuple[str, str]:
    """Clean and sanity-check an App Password before anything touches the network.

    Google displays App Passwords as "abcd efgh ijkl mnop" and people paste
    them with the spaces; Gmail then answers with a bare AUTHENTICATIONFAILED
    that says nothing about why. Stripping and length-checking here turns two
    common mistakes into messages that name themselves, and saves a pointless
    round trip to Google for input that cannot possibly work.
    """
    password = password.replace(" ", "").strip()
    username = username.strip()
    if not username or "@" not in username:
        raise ValueError(f"{username!r} does not look like an email address")
    # if len(password) != 16:
    #     raise ValueError(
    #         f"an App Password is 16 characters; this one is {len(password)}. You want "
    #         "the password generated at myaccount.google.com/apppasswords, not your "
    #         "normal Google password."
    #     )
    return username, password


def save_imap_credentials(
    username: str, password: str, settings: Settings | None = None
) -> Path:
    """Store an IMAP username and App Password, 0600."""
    settings = settings or get_settings()
    username, password = normalize_imap_credentials(username, password)

    path = _imap_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"username": username, "password": password}), encoding="utf-8"
    )
    _chmod_600(path)
    log.info("stored IMAP app-password credentials for %s", username)
    return path


def imap_credentials(settings: Settings | None = None) -> tuple[str, str] | None:
    settings = settings or get_settings()
    path = _imap_path(settings)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["username"], data["password"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def forget_imap_credentials(settings: Settings | None = None) -> None:
    _imap_path(settings or get_settings()).unlink(missing_ok=True)


def verify_imap(username: str, password: str, settings: Settings | None = None) -> str:
    """Log in for real and report what the mailbox contains.

    Storing credentials that do not work is worse than refusing them: the
    failure resurfaces later inside a sync, where it looks like a sync bug.
    """
    settings = settings or get_settings()
    from imapclient import IMAPClient

    with IMAPClient(settings.imap_host, port=settings.imap_port, ssl=True) as client:
        client.login(username, password.replace(" ", ""))
        # Read-only: this check must not touch \Seen flags.
        folder = settings.imap_folder
        status = client.select_folder(folder, readonly=True)
        return f"{status[b'EXISTS']} messages in {folder}"


def save_client_secrets(raw: bytes, settings: Settings | None = None) -> Path:
    """Store an uploaded OAuth client secrets file, so the dashboard can take it.

    Validated rather than trusted: a wrong file here fails much later, inside
    the consent flow, with an error that says nothing about which file was
    wrong. The two accepted shapes are Google's own -- "installed" for a
    Desktop app client, "web" for a Web application one.
    """
    settings = settings or get_settings()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not valid JSON: {exc}") from exc

    kind = next((k for k in ("installed", "web") if k in parsed), None)
    if kind is None:
        raise ValueError(
            "this is not an OAuth client secrets file -- expected a top-level "
            '"installed" or "web" key. Downloading the wrong thing from Google '
            "Cloud Console is easy; you want the client's JSON, not a service "
            "account key."
        )
    if kind == "web":
        # It can still work, but the loopback consent flow below is built for
        # the desktop client type and this is the likelier mistake.
        log.warning("client secrets are a 'web' client; a Desktop app client is expected")
    if not parsed[kind].get("client_id"):
        raise ValueError("client secrets have no client_id")

    path = _credentials_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    _chmod_600(path)
    log.info("stored OAuth client secrets at %s (%s client)", path, kind)
    return path


def has_client_secrets(settings: Settings | None = None) -> bool:
    return _credentials_path(settings or get_settings()).exists()


def _write_token(path: Path, creds: Credentials) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(creds.to_json(), encoding="utf-8")
    _chmod_600(tmp)
    tmp.replace(path)
    log.info("stored refresh token at %s", path)


def _chmod_600(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except (OSError, NotImplementedError):  # pragma: no cover - Windows dev boxes
        log.debug("could not chmod 0600 %s on this platform", path)


def account_email(settings: Settings | None = None) -> str | None:
    """The address these credentials belong to, if it was recorded at consent."""
    settings = settings or get_settings()
    path = _token_path(settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("account") or None
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:  # pragma: no cover - operator entry point
    logging.basicConfig(level="INFO")
    creds = run_consent_flow()
    print(f"connected; scopes={creds.scopes}")


if __name__ == "__main__":  # pragma: no cover
    main()
