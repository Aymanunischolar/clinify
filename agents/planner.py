"""Planner agent node: decomposes the user request into search queries."""
from __future__ import annotations

from agents.llm import get_llm_client
from agents.prompts import PLANNER_SCHEMA_HINT, PLANNER_SYSTEM_PROMPT_V1
from agents.schemas import PlannerOutput


def run(user_input: str) -> PlannerOutput:
    llm = get_llm_client()
    raw = llm.structured_completion(
        system_prompt=PLANNER_SYSTEM_PROMPT_V1,
        user_prompt=user_input,
        schema_hint=PLANNER_SCHEMA_HINT,
    )
    try:
        return PlannerOutput(**raw)
    except Exception:
        return PlannerOutput(sub_questions=[user_input], search_queries=[user_input])
