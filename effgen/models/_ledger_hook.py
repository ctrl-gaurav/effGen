"""The two points inside an adapter a run's ledger needs to see.

An adapter spends time on its own work around a provider request — building the
request, parsing the answer, pricing it — and that time is effGen's, not the
model's. An adapter marks the request itself with :func:`provider_request`, and
sleeps between its own retries with :func:`backoff_sleep`, so the ledger of the
run in progress (:mod:`effgen.core.ledger`) counts only those as model time.

Both do nothing when no run is in progress, and neither imports the agent
package: a ledger can only be active once :mod:`effgen.core.ledger` has been
imported, so this module looks it up in :data:`sys.modules` rather than
importing it.
"""

from __future__ import annotations

import contextlib
import sys
import time
from typing import Any

_LEDGER = "effgen.core.ledger"


def provider_request() -> Any:
    """A context manager around one request to the provider's API."""
    ledger = sys.modules.get(_LEDGER)
    if ledger is None:
        return contextlib.nullcontext()
    return ledger.provider_request()


def backoff_sleep(seconds: float) -> None:
    """Sleep before retrying a provider request; the ledger counts it as model wait."""
    ledger = sys.modules.get(_LEDGER)
    if ledger is None:
        time.sleep(seconds)
        return
    ledger.backoff_sleep(time.sleep, seconds)
