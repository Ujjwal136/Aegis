from __future__ import annotations

from fastapi import FastAPI, HTTPException

from agents import BankingDB, LLMAgent, ManagingAgent
from config import settings
from firewall.interceptor import Interceptor
from firewall.redactor import Redactor
from firewall.sentinel import Sentinel
from firewall.weilchain import Weilchain
from models.schemas import (
    ChatRequest,
    ChatResponse,
    FirewallEgressRequest,
    FirewallEgressResponse,
    FirewallIngressRequest,
    FirewallIngressResponse,
    HealthResponse,
)


app = FastAPI(title=settings.app_name)

llm_agent = LLMAgent()
db = BankingDB()
managing_agent = ManagingAgent(db, llm_agent)
sentinel = Sentinel()
redactor = Redactor()
weilchain = Weilchain()
interceptor = Interceptor(sentinel, redactor, weilchain)

sentinel_loaded = sentinel.load()
redactor_loaded = redactor.load()


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    ingress = interceptor.ingress(payload.message, payload.session_id)
    if ingress["verdict"] == "BLOCKED":
        return ChatResponse(
            trace_id=ingress["trace_id"],
            verdict="BLOCKED",
            response=llm_agent.handle_blocked(),
            redactions=[],
        )

    raw_db = managing_agent.execute_planned_query(ingress["sanitized_prompt"])
    egress = interceptor.egress(ingress["trace_id"], payload.session_id, str(raw_db))
    synthesized = llm_agent.synthesize(egress["sanitized_payload"], payload.message)

    return ChatResponse(
        trace_id=ingress["trace_id"],
        verdict=egress["verdict"],
        response=synthesized,
        redactions=egress["redactions"],
    )


@app.post("/api/v1/firewall/ingress", response_model=FirewallIngressResponse)
def ingress(payload: FirewallIngressRequest) -> FirewallIngressResponse:
    result = interceptor.ingress(payload.prompt, payload.session_id)
    return FirewallIngressResponse(**result)


@app.post("/api/v1/firewall/egress", response_model=FirewallEgressResponse)
def egress(payload: FirewallEgressRequest) -> FirewallEgressResponse:
    result = interceptor.egress(payload.trace_id, payload.session_id, str(payload.payload))
    return FirewallEgressResponse(**result)


@app.get("/api/v1/audit/ledger")
def ledger() -> list[dict]:
    return weilchain.get_all()


@app.get("/api/v1/audit/verify/{trace_id}")
def verify(trace_id: str) -> dict:
    entries = [entry for entry in weilchain.get_all() if entry["trace_id"] == trace_id]
    if not entries:
        raise HTTPException(status_code=404, detail="trace_id not found")
    return {"trace_id": trace_id, "valid": all(weilchain.verify(entry) for entry in entries)}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", sentinel_loaded=sentinel_loaded, redactor_loaded=redactor_loaded)
