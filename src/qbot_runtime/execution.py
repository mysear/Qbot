from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol
import uuid

from .schemas import OrderResult


class ExecutionClient(Protocol):
    capability_id: str
    def submit(self, intent: dict[str, Any]) -> OrderResult: ...


@dataclass
class PaperExecutionClient:
    capability_id: str
    config: dict[str, Any]

    def submit(self, intent: dict[str, Any]) -> OrderResult:
        required = {"symbol", "side", "quantity", "idempotency_key"}
        missing = required-set(intent)
        if missing: raise ValueError(f"Order intent is missing: {', '.join(sorted(missing))}")
        if intent["side"] not in {"buy", "sell"} or float(intent["quantity"]) <= 0:
            raise ValueError("Order side or quantity is invalid")
        return OrderResult(str(uuid.uuid5(uuid.NAMESPACE_URL, str(intent["idempotency_key"]))), "accepted", {"paper": True, "intent": dict(intent)})


@dataclass
class DelegatingExecutionClient:
    capability_id: str
    submitter: Callable[[dict[str, Any]], dict[str, Any]]

    def submit(self, intent: dict[str, Any]) -> OrderResult:
        response = self.submitter(dict(intent))
        return OrderResult(str(response["order_id"]), str(response["status"]), dict(response.get("metadata", {})))


def create_execution_client(capability_id: str, config: dict[str, Any] | None = None) -> ExecutionClient:
    options = dict(config or {})
    if capability_id == "qbot.paper": return PaperExecutionClient(capability_id, options)
    submitter = options.get("submitter")
    if capability_id.startswith("qbot.execution.") and callable(submitter): return DelegatingExecutionClient(capability_id, submitter)
    raise ValueError(f"Unknown or unconfigured Qbot execution client: {capability_id}")
