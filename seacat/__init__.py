"""SeaCat API client. Typed decisions over HTTP, with no dependencies beyond the standard library.

    from seacat import SeaCat, category, scale, yes_no

    sc = SeaCat()  # reads SEACAT_API_KEY
    d = sc.decide(
        "Budget is approved and we need this live before November.",
        {
            "stage": category("How far along is this lead in buying?",
                              {"researching": "No timeline yet", "ready": "Budget approved"}),
            "fit": scale("How well does this match our target customer?", ["Poor", "Partial", "Strong"]),
            "wants_pricing": yes_no("Does the message ask about prices or plans?"),
        },
    )
    d["stage"].answer          # 'ready'
    d["stage"].probability     # 0.91
    d["fit"].mean              # 1.8
    d["wants_pricing"].is_yes  # True
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SeaCat",
    "AsyncSeaCat",
    "category",
    "scale",
    "yes_no",
    "Answer",
    "Decision",
    "Usage",
    "Model",
    "SeaCatError",
    "APIError",
    "AuthenticationError",
    "OutOfCredits",
    "AccessPending",
    "InvalidRequest",
    "NotFound",
    "RateLimited",
    "ModelUnavailable",
    "ServerError",
    "TransportError",
    "Timeout",
]

__version__ = "0.1.0"

DEFAULT_BASE_URL = "https://seacat.dev"
# A queued request is redirected to a result URL that waits about a minute per hop, so one HTTP request should
# never need more than this, and the whole call is bounded by `timeout` instead.
MAX_REQUEST_S = 120.0


# --- Questions -------------------------------------------------------------------------------------------------
# Small builders: a question is a plain dict, so a hand-written one works just as well.


def category(text: str, options: Mapping[str, str] | Sequence[str]) -> dict:
    """Pick one of 2 to 26 options. `options` is `{option: description}` or `[option, ...]`."""
    return {"type": "category", "text": text, "options": _options(options)}


def scale(text: str, levels: Sequence[str]) -> dict:
    """Rate on 2 to 26 ordered levels, lowest first."""
    return {"type": "scale", "text": text, "options": list(levels)}


def yes_no(text: str) -> dict:
    """Is a statement true?"""
    return {"type": "yes_no", "text": text}


def _options(options):
    return dict(options) if isinstance(options, Mapping) else list(options)


# --- Answers ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    """One question's answer: the most likely option, every option's probability, and how certain the model is."""

    type: str
    answer: str
    probabilities: dict[str, float]
    certainty: float
    mean: float | None = None  # scale questions only: the expected level, where 0 is the lowest

    @property
    def probability(self) -> float:
        """The chosen option's probability."""
        return self.probabilities[self.answer]

    @property
    def is_yes(self) -> bool:
        """True when a yes_no question answered `yes`."""
        return self.answer == "yes"

    def confident(self, threshold: float = 0.9) -> bool:
        """Whether `certainty` reaches `threshold`. Tune the threshold on your own data."""
        return self.certainty >= threshold

    def __str__(self) -> str:
        return self.answer

    @classmethod
    def _from(cls, data: dict) -> "Answer":
        return cls(
            type=data["type"],
            answer=data["answer"],
            probabilities=data["probabilities"],
            certainty=data["certainty"],
            mean=data.get("mean"),
        )


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class Decision(Mapping):
    """The answers to one request, keyed by question name: `d["stage"].answer`, or `for name, a in d.items()`."""

    model: str
    answers: dict[str, Answer]
    usage: Usage
    timing: dict[str, float] = field(repr=False, default_factory=dict)
    raw: dict = field(repr=False, default_factory=dict)

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.answers)

    def __len__(self) -> int:
        return len(self.answers)

    @classmethod
    def _from(cls, data: dict, timing: dict[str, float]) -> "Decision":
        return cls(
            model=data["model"],
            answers={name: Answer._from(a) for name, a in data["answers"].items()},
            usage=Usage(**data["usage"]),
            timing=timing,
            raw=data,
        )


@dataclass(frozen=True)
class Model:
    name: str
    price_per_mtok_usd: float


# --- Errors ----------------------------------------------------------------------------------------------------


class SeaCatError(Exception):
    """Base class for everything this client raises."""


class APIError(SeaCatError):
    """The API answered with an error. `status` is the HTTP status and `detail` what the API said."""

    def __init__(self, status: int, detail: str, retry_after: float | None = None):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """401: the API key is missing, invalid or revoked."""


class OutOfCredits(APIError):
    """402: the account is out of credits."""


class AccessPending(APIError):
    """403: the account is on the waitlist."""


class InvalidRequest(APIError):
    """400 or 422: the request needs fixing. Retrying it unchanged will fail again."""


class NotFound(APIError):
    """404: no result with this ID for this key, or it is over an hour old."""


class RateLimited(APIError):
    """429: over this key's requests per minute or requests in progress. Wait `retry_after` seconds."""


class ModelUnavailable(APIError):
    """503: the model couldn't be reached. Wait `retry_after` seconds."""


class ServerError(APIError):
    """An unexpected 5xx."""


class TransportError(SeaCatError):
    """The request never got an answer: DNS, connection or TLS failure."""


class Timeout(SeaCatError):
    """The call ran past `timeout`.

    A queued request keeps being worked on after this, and is charged once whether or not its answer is collected.
    When `result_url` is set, pass it to `SeaCat.result()` later to collect the answer, for up to an hour.
    """

    def __init__(self, message: str, result_url: str | None = None):
        super().__init__(message)
        self.result_url = result_url


_BY_STATUS = {
    400: InvalidRequest,
    401: AuthenticationError,
    402: OutOfCredits,
    403: AccessPending,
    404: NotFound,
    422: InvalidRequest,
    429: RateLimited,
    503: ModelUnavailable,
}


# --- Client ----------------------------------------------------------------------------------------------------


class SeaCat:
    """A client for one SeaCat server.

    api_key: your key. Defaults to `SEACAT_API_KEY`.
    base_url: the server. Defaults to `SEACAT_BASE_URL`, then https://seacat.dev.
    timeout: seconds for a whole call, redirects and retries included. The GPU scales to zero, so the first
        request after an idle spell waits a minute or two for it to start.
    retries: how many times to retry a timeout, a connection failure, a 429 or a 5xx. `Retry-After` is honoured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 300.0,
        retries: int = 2,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("SEACAT_API_KEY", "")
        self.base_url = (base_url or os.environ.get("SEACAT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._opener = urllib.request.build_opener(_NoRedirect)

    def decide(
        self,
        state: str | dict | list,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> Decision:
        """Answer every question about `state`. Each question is answered on its own, from the state and its own
        text, and every answer is one of your options.

        Raises an `APIError` subclass for an error from the API, `TransportError` if it couldn't be reached, and
        `Timeout` if the answer didn't arrive within `timeout`.
        """
        body: dict[str, Any] = {"state": state, "questions": dict(questions)}
        if model is not None:
            body["model"] = model
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        status, data, location, timing = self._request("POST", "/v1/decide", body, deadline)
        if status == 303:
            data, timing = self._collect(location, deadline)
        return Decision._from(data, timing)

    def result(self, result_url: str, *, timeout: float | None = None) -> Decision:
        """Collect the answer to a call that ran past its timeout, using `Timeout.result_url`.

        Only the key that made the request can collect it, for up to an hour, and the request is charged once
        however many times its result is fetched.
        """
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        return Decision._from(*self._collect(result_url, deadline))

    def models(self) -> list[Model]:
        """The model this server runs and its price. No API key needed."""
        _, data, *_ = self._request("GET", "/v1/models", None, time.monotonic() + self.timeout)
        return [Model(**m) for m in data["models"]]

    # A queued request is redirected to a result URL, which waits about a minute and then redirects to itself
    # until the answer is ready.
    def _collect(self, url: str, deadline: float) -> tuple[dict, dict[str, float]]:
        while True:
            status, data, location, timing = self._request("GET", url, None, deadline, result_url=url)
            if status != 303:
                return data, timing
            url = location

    def _request(self, method, url, body, deadline, result_url=None):
        url = url if url.startswith("http") else self.base_url + url
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Accept": "application/json", "User-Agent": f"seacat-python/{__version__}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.retries + 1):
            left = deadline - time.monotonic()
            if left <= 0:
                raise Timeout(f"No answer within the timeout while calling {method} {url}.", result_url)
            request = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self._opener.open(request, timeout=min(left, MAX_REQUEST_S)) as response:
                    return response.status, json.load(response), None, _timing(response.headers)
            except urllib.error.HTTPError as e:
                if e.status == 303:  # queued: the answer is at the result URL, which we follow ourselves
                    return 303, None, urllib.parse.urljoin(url, e.headers["Location"]), {}
                error = _api_error(e)
                if attempt == self.retries or not _retriable(e.status):
                    raise error from None
                wait = error.retry_after
            except (TimeoutError, urllib.error.URLError, OSError) as e:
                if attempt == self.retries:
                    if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
                        raise Timeout(f"No answer within the timeout while calling {method} {url}.", result_url)
                    raise TransportError(f"Could not reach {url}: {e}") from None
                wait = None
            time.sleep(min(_backoff(attempt, wait), max(0.0, deadline - time.monotonic())))
        raise AssertionError("unreachable")


class AsyncSeaCat:
    """The same client for async code. Every call runs the blocking one in a worker thread, so it doesn't hold
    up the event loop; a thread per in-flight request is a fair trade for having no dependencies.

        sc = AsyncSeaCat()
        d = await sc.decide(state, questions)
    """

    def __init__(self, api_key: str | None = None, **kwargs):
        self.sync = SeaCat(api_key, **kwargs)

    async def decide(self, state, questions, *, model=None, timeout=None) -> Decision:
        return await asyncio.to_thread(self.sync.decide, state, questions, model=model, timeout=timeout)

    async def result(self, result_url: str, *, timeout=None) -> Decision:
        return await asyncio.to_thread(self.sync.result, result_url, timeout=timeout)

    async def models(self) -> list[Model]:
        return await asyncio.to_thread(self.sync.models)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Hands the 303 a queued request gets back to `_request` instead of following it: the result URL redirects to
    itself while it waits, which urllib would give up on as a redirect loop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _timing(headers) -> dict[str, float]:
    """`Server-Timing` as {name: milliseconds}: where the server spent the request, for debugging."""
    timing = {}
    for part in (headers.get("Server-Timing") or "").split(","):
        name, _, rest = part.strip().partition(";dur=")
        try:
            timing[name] = float(rest)
        except ValueError:
            continue
    return timing


def _retriable(status: int) -> bool:
    return status == 429 or status >= 500


def _backoff(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return retry_after
    return min(2**attempt, 8) * (0.5 + random.random())  # jitter, so retries from many workers don't line up


def _api_error(e: urllib.error.HTTPError) -> APIError:
    retry_after = e.headers.get("Retry-After") if e.headers else None
    try:
        retry_after = float(retry_after) if retry_after else None
    except ValueError:  # the header may be an HTTP date, which we don't parse
        retry_after = None
    return _BY_STATUS.get(e.status, ServerError)(e.status, _detail(e), retry_after)


def _detail(e: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(e.read()).get("detail", "")
    except Exception:
        return e.reason or "request failed"
    if isinstance(detail, list):  # a 422 lists the fields that failed validation
        return "; ".join(f"{'.'.join(str(p) for p in d.get('loc', ()))}: {d.get('msg', '')}" for d in detail)
    return str(detail) or (e.reason or "request failed")
