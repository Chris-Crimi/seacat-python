# SeaCat for Python

A client for the [SeaCat](https://seacat.dev) API: you send some state and a set of typed questions, and you get
back structured answers with probabilities and a certainty score your code can branch on. Standard library only,
no dependencies.

```bash
pip install seacat
```

```python
from seacat import SeaCat, category, scale, yes_no

sc = SeaCat()  # or SeaCat("tz_...")  — the default reads SEACAT_API_KEY

d = sc.decide(
    "Hi, I run operations at a 40-person logistics company. Budget is approved and we need "
    "something live before our peak season in November. Could you walk me through pricing for 25 seats?",
    {
        "stage": category(
            "How far along is this lead in buying?",
            {
                "researching": "Early research, no timeline or budget yet",
                "evaluating": "Comparing options, with a rough timeline",
                "ready": "Budget approved and a firm deadline",
            },
        ),
        "fit": scale(
            "How well does this company match our target customer: logistics or retail, 20 to 500 people?",
            ["Poor match", "Partial match", "Strong match"],
        ),
        "wants_pricing": yes_no("Does the message ask about prices or plans?"),
    },
)

if d["stage"].answer == "ready" and d["fit"].mean > 1.5:
    route_to_sales()
```

## Questions

`category(text, options)` picks one of 2 to 26 options, given as `{option: description}` or `[option, ...]`.
`scale(text, levels)` rates on 2 to 26 ordered levels, lowest first. `yes_no(text)` asks whether a statement is
true. Each builder returns a plain dict, so a hand-written one works just as well:

```python
sc.decide(state, {"is_spam": {"type": "yes_no", "text": "Is this comment spam?"}})
```

The state can be a string, or a dict or list, which is sent as JSON. It is read once per request and shared by
every question, so asking ten questions about one state costs far less than ten requests.

Each question sees only the state and its own text — not your other questions, and not their answers. Define any
term it can't guess, and precompute totals and counts into the state rather than asking for arithmetic.

## Answers

`decide()` returns a `Decision`: a mapping of question name to `Answer`, plus `model` and `usage`.

```python
d["stage"].answer         # 'ready' — always one of your options
d["stage"].probabilities  # {'researching': 0.02, 'evaluating': 0.07, 'ready': 0.91}
d["stage"].probability    # 0.91 — the chosen option's probability
d["stage"].certainty      # 0.77 — 1 when one option has all of it, 0 when it's split evenly
d["stage"].confident(0.8) # False
d["fit"].mean             # 1.75 — scale questions only: the expected level, where 0 is the lowest
d["wants_pricing"].is_yes # True
d.model, d.usage.input_tokens, d.usage.cost_usd
d.timing                  # Server-Timing as {name: ms}, for debugging: where the server spent the request
d.raw                     # the response exactly as the API sent it

for name, answer in d.items():
    print(name, answer.answer, answer.probability)
```

Certainty is a measure of how spread the probabilities are, not a promise of being right. Tune your thresholds on
your own data.

## Async

`AsyncSeaCat` takes the same arguments and has the same three methods, awaited. Each call runs the blocking
client in a worker thread, so it doesn't hold up the event loop — a thread per in-flight request is a fair trade
for having no dependencies.

```python
from seacat import AsyncSeaCat

sc = AsyncSeaCat()
first, second = await asyncio.gather(sc.decide(a, questions), sc.decide(b, questions))
```

## Errors

Everything raised inherits from `SeaCatError`. An error from the API is an `APIError` with `status`, `detail` and,
where the API sent one, `retry_after`:

| Class | Status | |
|---|---|---|
| `InvalidRequest` | 400, 422 | The request needs fixing: too long for the model's input limit, an unknown `model`, or a malformed question. Retrying it unchanged will fail again. |
| `AuthenticationError` | 401 | The key is missing, invalid or revoked. |
| `OutOfCredits` | 402 | Add credits on the dashboard. |
| `AccessPending` | 403 | The account is on the waitlist. |
| `NotFound` | 404 | No result with this ID for this key, or it is over an hour old. |
| `RateLimited` | 429 | Over this key's requests per minute, or requests in progress. |
| `ModelUnavailable` | 503 | The model couldn't be reached. |
| `ServerError` | 5xx | Unexpected. |
| `TransportError` | — | The request never got an answer: DNS, connection or TLS. |
| `Timeout` | — | No answer within `timeout`. |

A timeout, a connection failure, a `429` and a `5xx` are retried on their own (`retries=2` by default), waiting as
long as `Retry-After` says. The rest are raised straight away, because retrying them unchanged would fail again.

## Cold starts and slow requests

The GPU scales to zero, so the first request after an idle spell waits a minute or two for it to start. The API
answers a slow request with a redirect to a result URL, and the client follows it for you.

`timeout` (300 seconds by default) covers the whole call, redirects and retries included. When it runs out, the
`Timeout` carries the result URL, and the answer can still be collected for up to an hour, with the same key:

```python
try:
    d = sc.decide(state, questions, timeout=30)
except Timeout as e:
    save(e.result_url)     # ... later, in a worker:
    d = sc.result(e.result_url)
```

A queued request is charged once when it finishes, whether or not its answer is collected.

## Configuration

```python
SeaCat(
    api_key=None,     # defaults to SEACAT_API_KEY
    base_url=None,    # defaults to SEACAT_BASE_URL, then https://seacat.dev
    timeout=300.0,    # seconds for a whole call, redirects and retries included
    retries=2,        # retries for a timeout, a connection failure, a 429 or a 5xx
)
```

`sc.models()` returns the model this server runs and its price. It needs no key.

The client is thread-safe: `SeaCat` holds no per-request state, and every call makes its own connection. The
package ships a `py.typed` marker, so type checkers read its annotations.

## Tests

`pytest` in this directory, against a local HTTP server that answers the way the API does, including the
queued-result path. It needs nothing but pytest.

## License

MIT — see [LICENSE](LICENSE). The client is MIT so you can install, read and modify it freely. The SeaCat
service it calls is a separate, proprietary product, governed by the [Terms](https://seacat.dev/terms).
