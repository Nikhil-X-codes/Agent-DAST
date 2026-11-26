"""FastAPI HTTP wrapper for hosting CustomerSupportAgent as a local HTTP target."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from agentshield.targets.support.agent import CustomerSupportAgent


class HTTPInvokeRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    authenticated_subject: str | None = None


def create_support_http_app(agent: CustomerSupportAgent, *, expose_telemetry: bool = True) -> FastAPI:
    """Create a FastAPI app exposing the given CustomerSupportAgent over HTTP."""
    app = FastAPI(title="AgentShield Support Target HTTP App")

    @app.get("/capabilities")
    def capabilities() -> dict[str, object]:
        if not expose_telemetry:
            return {
                "tools": [],
                "declared_permissions": [],
                "telemetry_available": False,
                "session_handling": "stateless",
            }
        return {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                    "permissions": [f"variant:{agent.variant}"],
                }
                for t in agent.executor.tool_definitions
            ],
            "declared_permissions": [f"authenticated_customer:{agent.authenticated_customer_id}"],
            "telemetry_available": True,
            "session_handling": "session_bound",
        }

    @app.post("/invoke")
    def invoke(body: HTTPInvokeRequest) -> dict[str, object]:
        if body.authenticated_subject:
            agent.authenticated_customer_id = body.authenticated_subject
            agent.executor.authenticated_customer_id = body.authenticated_subject

        result = agent.run(body.prompt)
        payload: dict[str, object] = {
            "response_text": result.final_response,
            "provider": result.provider,
            "model": result.model,
            "usage": result.usage.model_dump(),
        }
        if expose_telemetry:
            payload["tool_executions"] = [rec.model_dump() for rec in result.tool_executions]
            payload["llm_turns"] = [t.model_dump() for t in result.llm_turns]
            payload["variant"] = result.variant
        return payload

    return app
