"""Dropbox stealth TLS / HTTP-redirect proxy.

Listens on a single port (default ``8443``). Peeks the first byte of every
accepted TCP connection:

- TLS (``0x16``): terminates TLS and reverse-proxies authenticated HTTP to
  the upstream filebrowser. Unauthenticated requests get a bare-bones
  ``<input type=password>`` login page; a successful POST to ``/__auth``
  sets an HMAC-signed cookie and redirects to ``/``.
- Plain HTTP (any other byte): responds with ``301 Moved Permanently`` to
  ``https://<host>:<listen_port>/<path>`` so users hitting the wrong
  scheme on the right port get redirected instead of a TLS handshake error.

Stealth choices: no ``Server`` header, no body on errors, login page is a
single password input with no labels, no CSS, no title.

Configuration is loaded via :class:`autosre.dropbox.config.DropboxConfig`;
no module-level constants other than the wire-protocol literals.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import secrets
import socket
import ssl
import time
import urllib.parse

from autosre.dropbox.config import DropboxConfig

# Wire-protocol literals (intentionally not configurable)
COOKIE_NAME = "dbx"
LOGIN_PATH = "/__auth"
LOGIN_HTML: bytes = (
    b"<!doctype html><title></title>"
    b"<form method=POST action=" + LOGIN_PATH.encode() + b">"
    b"<input type=password name=p autofocus autocomplete=off>"
    b"</form>"
)
HOP_BY_HOP_HEADERS: tuple[bytes, ...] = (
    b"connection:",
    b"keep-alive:",
    b"proxy-connection:",
    b"te:",
    b"trailer:",
    b"transfer-encoding:",
    b"upgrade:",
)
MAX_REQUEST_BYTES = 131072


# ---------------------------------------------------------------------------
# Credential + secret helpers
# ---------------------------------------------------------------------------


def _load_password(config: DropboxConfig) -> str:
    """Read the admin password file. Returns empty string if missing."""
    try:
        text = config.password_file.read_text()
    except FileNotFoundError:
        return ""
    for line in text.splitlines():
        if line.startswith("ADMIN_PW="):
            return line.split("=", 1)[1].strip()
        # Allow a bare-password file (one line, no key=value) for convenience.
        stripped = line.strip()
        if stripped and not line.startswith("#"):
            return stripped
    return ""


def _load_secret(config: DropboxConfig) -> bytes:
    """Read or initialize the HMAC signing secret (mode 0600)."""
    secret_path = config.secret_file
    if not secret_path.exists():
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        secret_path.write_bytes(secrets.token_bytes(32))
        secret_path.chmod(0o600)
    return secret_path.read_bytes()


def _sign(secret: bytes, exp: int) -> str:
    sig = hmac.new(secret, str(exp).encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def _verify(secret: bytes, token: str) -> bool:
    try:
        exp_s, sig = token.split(".", 1)
        exp = int(exp_s)
    except (ValueError, AttributeError):
        return False
    if exp < int(time.time()):
        return False
    want = hmac.new(secret, exp_s.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, sig)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _build_response(
    status: str,
    body: bytes = b"",
    extra_headers: list[tuple[str, str]] | None = None,
    content_type: str = "text/html; charset=utf-8",
) -> bytes:
    headers = [
        f"HTTP/1.1 {status}",
        f"Content-Length: {len(body)}",
        f"Content-Type: {content_type}",
        "Connection: close",
        "Cache-Control: no-store",
        "X-Content-Type-Options: nosniff",
    ]
    for key, value in extra_headers or []:
        headers.append(f"{key}: {value}")
    return ("\r\n".join(headers) + "\r\n\r\n").encode() + body


async def _read_http_request(reader: asyncio.StreamReader) -> tuple[bytes, bytes] | None:
    buf = bytearray()
    while b"\r\n\r\n" not in buf:
        chunk = await reader.read(4096)
        if not chunk:
            return None
        buf.extend(chunk)
        if len(buf) > MAX_REQUEST_BYTES:
            return None
    head, _, rest = buf.partition(b"\r\n\r\n")
    content_length = 0
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            try:
                content_length = int(line.split(b":", 1)[1].strip())
            except ValueError:
                content_length = 0
            break
    body = bytearray(rest)
    while len(body) < content_length:
        chunk = await reader.read(content_length - len(body))
        if not chunk:
            break
        body.extend(chunk)
    return bytes(head), bytes(body[:content_length] if content_length else body)


def _parse_cookies(head: bytes) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for line in head.split(b"\r\n")[1:]:
        if line.lower().startswith(b"cookie:"):
            decoded = line.split(b":", 1)[1].decode("latin-1", "replace")
            for part in decoded.split(";"):
                if "=" in part:
                    key, value = part.strip().split("=", 1)
                    cookies[key] = value
    return cookies


# ---------------------------------------------------------------------------
# Connection plumbing
# ---------------------------------------------------------------------------


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError, OSError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


async def _handle_tls_http(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    config: DropboxConfig,
    secret: bytes,
) -> None:
    request = await _read_http_request(reader)
    if request is None:
        writer.close()
        return
    head, body = request
    try:
        first_line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        method, path, _ = first_line.split(" ", 2)
    except (UnicodeDecodeError, ValueError):
        writer.close()
        return

    if method == "POST" and path == LOGIN_PATH:
        await _handle_login(writer, body, config, secret)
        return

    cookies = _parse_cookies(head)
    if _verify(secret, cookies.get(COOKIE_NAME, "")):
        await _proxy_to_upstream(reader, writer, head, body, config)
        return

    writer.write(_build_response("200 OK", LOGIN_HTML))
    await writer.drain()
    writer.close()


async def _handle_login(
    writer: asyncio.StreamWriter,
    body: bytes,
    config: DropboxConfig,
    secret: bytes,
) -> None:
    submitted = ""
    for pair in body.decode("latin-1", "replace").split("&"):
        if pair.startswith("p="):
            submitted = urllib.parse.unquote_plus(pair[2:])
            break
    password = _load_password(config)
    if password and hmac.compare_digest(submitted, password):
        token = _sign(secret, int(time.time()) + config.cookie_ttl_seconds)
        cookie_value = (
            f"{COOKIE_NAME}={token}; Path=/; HttpOnly; Secure; SameSite=Lax; "
            f"Max-Age={config.cookie_ttl_seconds}"
        )
        writer.write(
            _build_response(
                "303 See Other",
                b"",
                [("Location", "/"), ("Set-Cookie", cookie_value)],
            )
        )
    else:
        writer.write(_build_response("401 Unauthorized", LOGIN_HTML))
    await writer.drain()
    writer.close()


async def _proxy_to_upstream(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    head: bytes,
    body: bytes,
    config: DropboxConfig,
) -> None:
    try:
        upstream_reader, upstream_writer = await asyncio.open_connection(*config.upstream)
    except OSError:
        writer.write(_build_response("502 Bad Gateway"))
        await writer.drain()
        writer.close()
        return

    lines = head.split(b"\r\n")
    kept = [lines[0]]
    for line in lines[1:]:
        if line.lower().startswith(HOP_BY_HOP_HEADERS):
            continue
        kept.append(line)
    kept.append(b"Connection: close")
    new_head = b"\r\n".join(kept)
    upstream_writer.write(new_head + b"\r\n\r\n" + body)
    await upstream_writer.drain()
    await asyncio.gather(_pipe(reader, upstream_writer), _pipe(upstream_reader, writer))


async def _handle_plain_redirect(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    first_byte: bytes,
    config: DropboxConfig,
) -> None:
    buf = bytearray(first_byte)
    while b"\r\n\r\n" not in buf and len(buf) < 8192:
        chunk = await reader.read(1024)
        if not chunk:
            break
        buf.extend(chunk)
    header = buf.decode("latin-1", "replace")
    lines = header.split("\r\n")
    request_path = "/"
    host = "localhost"
    if lines:
        parts = lines[0].split(" ")
        if len(parts) >= 2:
            request_path = parts[1]
    for line in lines[1:]:
        if line.lower().startswith("host:"):
            host = line.split(":", 1)[1].strip().split(":")[0]
            break
    response = (
        f"HTTP/1.1 301 Moved Permanently\r\n"
        f"Location: https://{host}:{config.listen_port}{request_path}\r\n"
        f"Content-Length: 0\r\nConnection: close\r\n\r\n"
    ).encode()
    writer.write(response)
    await writer.drain()
    writer.close()


# ---------------------------------------------------------------------------
# Peek + dispatch
# ---------------------------------------------------------------------------


async def _peek_first_byte(sock: socket.socket) -> bytes:
    loop = asyncio.get_running_loop()
    sock.setblocking(False)  # noqa: FBT003 — socket API takes a positional bool
    fut: asyncio.Future[bytes] = loop.create_future()

    def _on_readable() -> None:
        try:
            data = sock.recv(1, socket.MSG_PEEK)
        except (BlockingIOError, InterruptedError):
            return
        except OSError as exc:
            loop.remove_reader(sock.fileno())
            if not fut.done():
                fut.set_exception(exc)
            return
        loop.remove_reader(sock.fileno())
        if not fut.done():
            fut.set_result(data)

    loop.add_reader(sock.fileno(), _on_readable)
    try:
        return await asyncio.wait_for(fut, timeout=15)
    except (TimeoutError, OSError):
        with contextlib.suppress(KeyError, OSError):
            loop.remove_reader(sock.fileno())
        raise


async def _accept_and_dispatch(
    ssl_ctx: ssl.SSLContext,
    client_sock: socket.socket,
    config: DropboxConfig,
    secret: bytes,
) -> None:
    loop = asyncio.get_running_loop()
    try:
        peek = await _peek_first_byte(client_sock)
        if not peek:
            client_sock.close()
            return
        if peek[0] == 0x16:
            try:
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                transport, _ = await loop.connect_accepted_socket(
                    lambda: protocol,
                    client_sock,
                    ssl=ssl_ctx,
                    ssl_handshake_timeout=10,
                )
                writer = asyncio.StreamWriter(transport, protocol, reader, loop)
                await _handle_tls_http(reader, writer, config, secret)
            except (ssl.SSLError, OSError):
                with contextlib.suppress(OSError):
                    client_sock.close()
        else:
            try:
                reader = asyncio.StreamReader()
                protocol = asyncio.StreamReaderProtocol(reader)
                transport, _ = await loop.connect_accepted_socket(lambda: protocol, client_sock)
                writer = asyncio.StreamWriter(transport, protocol, reader, loop)
                first = await reader.read(1)
                await _handle_plain_redirect(reader, writer, first, config)
            except OSError:
                with contextlib.suppress(OSError):
                    client_sock.close()
    except (TimeoutError, OSError):
        with contextlib.suppress(OSError):
            client_sock.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(config: DropboxConfig) -> None:
    """Run the dropbox proxy until cancelled."""
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(str(config.cert_file), str(config.key_file))

    secret = _load_secret(config)

    loop = asyncio.get_running_loop()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((config.bind_addr, config.listen_port))
    server_sock.listen(256)
    server_sock.setblocking(False)  # noqa: FBT003 — socket API takes a positional bool

    # Each accept spawns a detached connection task. We deliberately don't
    # await or gc-track them — they self-terminate when the client closes,
    # and holding references would leak memory over the lifetime of the
    # server. The strong-ref guard that RUF006 asks for is unnecessary in
    # a forever-loop accept pattern.
    _pending: set[asyncio.Task[None]] = set()
    while True:
        client_sock, _addr = await loop.sock_accept(server_sock)
        task = loop.create_task(_accept_and_dispatch(ssl_ctx, client_sock, config, secret))
        _pending.add(task)
        task.add_done_callback(_pending.discard)


if __name__ == "__main__":
    asyncio.run(main(DropboxConfig.load()))
