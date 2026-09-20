"""Tests the client against a real HTTP server that answers like the API does."""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from seacat import (
    AsyncSeaCat,
    AuthenticationError,
    InvalidRequest,
    OutOfCredits,
    RateLimited,
    SeaCat,
    Timeout,
    TransportError,
    category,
    scale,
    yes_no,
)

ANSWERS = {
    "stage": {
        "type": "category",
        "answer": "ready",
        "probabilities": {"researching": 0.02, "evaluating": 0.07, "ready": 0.91},
        "certainty": 0.77,
    },
    "fit": {
        "type": "scale",
        "answer": "Strong",
        "probabilities": {"Poor": 0.05, "Partial": 0.15, "Strong": 0.8},
        "certainty": 0.5,
        "mean": 1.75,
    },
    "spam": {"type": "yes_no", "answer": "no", "probabilities": {"yes": 0.1, "no": 0.9}, "certainty": 0.53},
}
BODY = {"model": "seacat-1", "answers": ANSWERS, "usage": {"input_tokens": 1000, "cost_usd": 0.0002}}


class Server:
    """A stand-in API. `plan` is the responses it gives, oldest first; `requests` is what it saw."""

    def __init__(self):
        self.requests = []
        self.plan = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length).decode() if length else ""
                server.requests.append(
                    {"method": self.command, "path": self.path, "auth": self.headers.get("Authorization"), "body": body}
                )
                if self.path == "/v1/models":
                    return self.send(200, {"models": [{"name": "seacat-1", "price_per_mtok_usd": 0.2}]})
                step = server.plan.pop(0) if server.plan else {"status": 200, "body": BODY}
                if step["status"] == 303:
                    return self.send(303, {"status": "queued"}, {"Location": step["location"]})
                time.sleep(step.get("delay", 0))
                headers = {"Server-Timing": "auth;dur=0.1, model;dur=137.0, engine;dur=41.0, charge;dur=21.0"}
                self.send(step["status"], step["body"], step.get("headers", headers))

            do_GET = do_POST = _handle

            def send(self, status, body, headers={}):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def client(self, **kwargs):
        self.requests.clear()
        self.plan.clear()
        return SeaCat("sk-test", base_url=self.url, **{"retries": 0, **kwargs})


@pytest.fixture(scope="module")
def server():
    s = Server()
    yield s
    s.http.shutdown()


def spam():
    return {"spam": yes_no("Is this spam?")}


def test_sends_the_key_and_questions_and_reads_the_answers(server):
    sc = server.client()
    d = sc.decide(
        "A lead wrote in.",
        {
            "stage": category("How far along?", {"researching": "Early", "evaluating": "Comparing", "ready": "Approved"}),
            "fit": scale("How well does it match?", ["Poor", "Partial", "Strong"]),
            "spam": yes_no("Is this spam?"),
        },
    )

    sent = json.loads(server.requests[0]["body"])
    assert server.requests[0]["method"] == "POST"
    assert server.requests[0]["path"] == "/v1/decide"
    assert server.requests[0]["auth"] == "Bearer sk-test"
    assert sent["state"] == "A lead wrote in."
    assert sent["questions"]["fit"] == {
        "type": "scale",
        "text": "How well does it match?",
        "options": ["Poor", "Partial", "Strong"],
    }
    assert sent["questions"]["spam"] == {"type": "yes_no", "text": "Is this spam?"}
    assert "model" not in sent

    assert d.model == "seacat-1"
    assert d["stage"].answer == "ready"
    assert d["stage"].probability == 0.91
    assert d["fit"].mean == 1.75
    assert d["spam"].is_yes is False
    assert d["stage"].confident(0.7) is True
    assert d["stage"].confident() is False
    assert d.usage.input_tokens == 1000
    assert d.usage.cost_usd == 0.0002
    assert d.timing == {"auth": 0.1, "model": 137.0, "engine": 41.0, "charge": 21.0}  # Server-Timing, for debugging
    assert d.raw == BODY
    # It reads like a mapping of answers.
    assert sorted(d) == ["fit", "spam", "stage"]
    assert len(d) == 3
    assert str(d["stage"]) == "ready"


def test_passes_model_when_given(server):
    sc = server.client()
    sc.decide("hi", spam(), model="latest")
    assert json.loads(server.requests[0]["body"])["model"] == "latest"


def test_serializes_a_json_state(server):
    sc = server.client()
    sc.decide({"amount": 42, "vendor": "Acme"}, spam())
    assert json.loads(server.requests[0]["body"])["state"] == {"amount": 42, "vendor": "Acme"}


def test_follows_the_redirect_a_queued_request_gets(server):
    sc = server.client()
    server.plan.extend(
        [
            {"status": 303, "location": "/v1/decide/result/fc-1"},
            {"status": 303, "location": "/v1/decide/result/fc-1"},  # the result URL redirects to itself while it waits
            {"status": 200, "body": BODY},
        ]
    )
    d = sc.decide("hi", spam())
    assert d["spam"].answer == "no"
    assert [f"{r['method']} {r['path']}" for r in server.requests] == [
        "POST /v1/decide",
        "GET /v1/decide/result/fc-1",
        "GET /v1/decide/result/fc-1",
    ]
    assert all(r["auth"] == "Bearer sk-test" for r in server.requests)


def test_collects_a_queued_result_later(server):
    sc = server.client()
    d = sc.result("/v1/decide/result/fc-2")
    assert server.requests[0]["path"] == "/v1/decide/result/fc-2"
    assert d["stage"].answer == "ready"


@pytest.mark.parametrize(
    "status,detail,expected",
    [
        (401, "Invalid or missing API key.", AuthenticationError),
        (402, "Out of credits.", OutOfCredits),
        (400, "State plus question is 40213 tokens.", InvalidRequest),
    ],
)
def test_raises_for_an_error(server, status, detail, expected):
    sc = server.client()
    server.plan.append({"status": status, "body": {"detail": detail}})
    with pytest.raises(expected) as e:
        sc.decide("hi", spam())
    assert e.value.status == status
    assert e.value.detail == detail
    assert len(server.requests) == 1  # not retried


def test_reads_a_422s_field_errors(server):
    sc = server.client()
    server.plan.append(
        {"status": 422, "body": {"detail": [{"loc": ["body", "questions", "spam", "type"], "msg": "unexpected value"}]}}
    )
    with pytest.raises(InvalidRequest) as e:
        sc.decide("hi", spam())
    assert e.value.detail == "body.questions.spam.type: unexpected value"


def test_retries_a_429_after_retry_after(server):
    sc = server.client(retries=2)
    server.plan.extend(
        [
            {"status": 429, "body": {"detail": "Too many requests."}, "headers": {"Retry-After": "0"}},
            {"status": 200, "body": BODY},
        ]
    )
    started = time.monotonic()
    assert sc.decide("hi", spam())["spam"].answer == "no"
    assert len(server.requests) == 2
    assert time.monotonic() - started < 1


def test_gives_up_after_the_last_retry(server):
    sc = server.client(retries=1)
    limited = {"status": 429, "body": {"detail": "Too many requests."}, "headers": {"Retry-After": "0"}}
    server.plan.extend([limited, dict(limited)])
    with pytest.raises(RateLimited) as e:
        sc.decide("hi", spam())
    assert e.value.retry_after == 0
    assert len(server.requests) == 2


def test_retries_a_500(server):
    sc = server.client(retries=1)
    server.plan.extend([{"status": 500, "body": {"detail": "boom"}}, {"status": 200, "body": BODY}])
    sc.decide("hi", spam())
    assert len(server.requests) == 2


def test_times_out_and_reports_where_the_answer_can_be_collected(server):
    sc = server.client(timeout=0.3)
    server.plan.extend(
        [{"status": 303, "location": "/v1/decide/result/fc-3"}, {"status": 200, "body": BODY, "delay": 2}]
    )
    with pytest.raises(Timeout) as e:
        sc.decide("hi", spam())
    assert e.value.result_url.endswith("/v1/decide/result/fc-3")


def test_raises_transport_error_when_the_server_cant_be_reached():
    sc = SeaCat("k", base_url="http://127.0.0.1:1", retries=0)
    with pytest.raises(TransportError):
        sc.decide("hi", spam())


def test_models_needs_no_key(server):
    server.requests.clear()
    sc = SeaCat("", base_url=server.url)
    models = sc.models()
    assert [(m.name, m.price_per_mtok_usd) for m in models] == [("seacat-1", 0.2)]
    assert server.requests[0]["auth"] is None


def test_reads_the_key_and_base_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("SEACAT_API_KEY", "sk-env")
    monkeypatch.setenv("SEACAT_BASE_URL", "https://example.test/")
    sc = SeaCat()
    assert sc.api_key == "sk-env"
    assert sc.base_url == "https://example.test"


def test_the_async_client_answers_the_same_way(server):
    server.requests.clear()
    server.plan.clear()
    sc = AsyncSeaCat("sk-test", base_url=server.url, retries=0)

    async def ask():
        return await asyncio.gather(sc.decide("hi", spam()), sc.decide("hi", spam()))

    first, second = asyncio.run(ask())
    assert first["spam"].answer == second["spam"].answer == "no"
    assert len(server.requests) == 2  # both ran without blocking the loop
