"""Multi-API Key Pool & LLM Manager for CareFlow.
Enables parallel agent execution without rate-limiting by managing a pool of Groq API keys
from .env (TRIAGE_API, REPORT_API, EVENT_API, INTAKE_API, GROQ_API_KEY, GROQ_API_KEYS, etc.).
Provides automatic round-robin rotation, agent-specific key routing, and failover.
"""
from __future__ import annotations

import itertools
import os
import re
import threading
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Automatically load environment variables from .env
load_dotenv()

# Direct agent-specific environment variable mappings
AGENT_ENV_VARS: Dict[str, List[str]] = {
    "triage": ["TRIAGE_API", "TRIAGE_API_KEY", "GROQ_API_KEY_TRIAGE"],
    "report": ["REPORT_API", "REPORT_API_KEY", "GROQ_API_KEY_REPORT"],
    "scheduling": ["EVENT_API", "SCHEDULING_API", "EVENT_API_KEY", "SCHEDULING_API_KEY", "GROQ_API_KEY_SCHEDULING"],
    "event": ["EVENT_API", "SCHEDULING_API", "EVENT_API_KEY"],
    "intake": ["INTAKE_API", "INTAKE_API_KEY", "GROQ_API_KEY_INTAKE"],
    "intake_extractor": ["INTAKE_API", "INTAKE_API_KEY"],
    "supervisor": ["SUPERVISOR_API", "INTAKE_API"],
    "memory_consolidation": ["MEMORY_API", "MEMORY_API_KEY", "CONSOLIDATION_API"],
}

# Deterministic, non-overlapping key slots for parallel agent execution
AGENT_KEY_SLOTS: Dict[str, int] = {
    "intake": 0,
    "intake_extractor": 0,
    "triage": 1,
    "report": 2,
    "scheduling": 3,
    "memory_consolidation": 4,
    "domain_classifier": 1,
    "supervisor": 0,
}


class APIKeyPool:
    def __init__(self):
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._counter = 0
        self.reload_keys()

    def reload_keys(self) -> List[str]:
        """Scans environment and .env for all Groq API keys."""
        with self._lock:
            load_dotenv()
            collected: List[str] = []

            # 1. Direct agent-specific keys: TRIAGE_API, REPORT_API, EVENT_API, INTAKE_API, etc.
            for agent, env_list in AGENT_ENV_VARS.items():
                for var_name in env_list:
                    val = os.environ.get(var_name, "").strip().strip("'\"")
                    if val and val not in collected:
                        collected.append(val)

            # 2. Comma-separated list: GROQ_API_KEYS="key1,key2,key3"
            multi = os.environ.get("GROQ_API_KEYS", "")
            if multi:
                for item in multi.split(","):
                    k = item.strip().strip("'\"")
                    if k and k not in collected:
                        collected.append(k)

            # 3. Primary key: GROQ_API_KEY="key"
            primary = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")
            if primary and primary not in collected:
                collected.append(primary)

            # 4. Numbered keys: GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
            for env_name, val in os.environ.items():
                if env_name.startswith("GROQ_API_KEY_") and val.strip():
                    k = val.strip().strip("'\"")
                    if k and k not in collected:
                        collected.append(k)

            # 5. Any other environment variable starting with gsk_
            for env_name, val in os.environ.items():
                if isinstance(val, str) and val.startswith("gsk_"):
                    k = val.strip().strip("'\"")
                    if k and k not in collected:
                        collected.append(k)

            self._keys = collected
            return list(self._keys)

    def get_keys(self) -> List[str]:
        if not self._keys:
            return self.reload_keys()
        return list(self._keys)

    def get_next_key(self) -> Optional[str]:
        """Round-robin retrieval for parallel agents."""
        keys = self.get_keys()
        if not keys:
            return None
        with self._lock:
            key = keys[self._counter % len(keys)]
            self._counter += 1
            return key

    def get_key_for_agent(self, agent_name: str) -> Optional[str]:
        """Provides a dedicated non-colliding key per agent domain for parallel execution.
        1. Checks for explicit agent environment variables (TRIAGE_API, REPORT_API, EVENT_API, INTAKE_API).
        2. If intake has no dedicated key, avoids colliding with triage or report in parallel execution.
        3. Falls back to slot-based mapping across the pooled keys.
        """
        keys = self.get_keys()
        if not keys:
            return None

        # 1. Check explicit agent env vars
        env_vars = AGENT_ENV_VARS.get(agent_name, [])
        for var in env_vars:
            val = os.environ.get(var, "").strip().strip("'\"")
            if val:
                return val

        # 2. Check if primary GROQ_API_KEY is available
        primary = os.environ.get("GROQ_API_KEY", "").strip().strip("'\"")
        if primary and primary in keys:
            return primary

        # 3. For intake running parallel to triage or report, prioritize non-colliding keys
        if agent_name in ("intake", "intake_extractor", "supervisor"):
            triage_key = os.environ.get("TRIAGE_API", "").strip().strip("'\"")
            report_key = os.environ.get("REPORT_API", "").strip().strip("'\"")
            event_key = os.environ.get("EVENT_API", "").strip().strip("'\"")
            
            # If EVENT_API is available, it is ideal for intake because scheduling never executes in parallel with intake
            if event_key and event_key in keys:
                return event_key
            for k in keys:
                if k not in (triage_key, report_key):
                    return k
            for k in keys:
                if k != triage_key:
                    return k

        # 4. Deterministic slot assignment
        slot = AGENT_KEY_SLOTS.get(agent_name, abs(hash(agent_name)))
        idx = slot % len(keys)
        return keys[idx]

    def get_failover_key(self, current_key: Optional[str] = None) -> Optional[str]:
        """Returns an alternate key from the pool if the current key hit a rate limit or failure."""
        keys = self.get_keys()
        if not keys:
            return None
        candidates = [k for k in keys if k != current_key]
        if candidates:
            return candidates[0]
        return self.get_next_key()

    def create_llm(
        self,
        agent_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 600,
        failover_from: Optional[str] = None,
    ) -> ChatGroq:
        """Instantiates a ChatGroq LLM configured with a key from the pool."""
        keys = self.get_keys()
        model = model_name or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

        if not keys:
            # Fallback to dummy or raise notice
            api_key = "missing_api_key"
        elif failover_from:
            api_key = self.get_failover_key(failover_from) or keys[0]
        elif agent_name:
            api_key = self.get_key_for_agent(agent_name) or keys[0]
        else:
            api_key = self.get_next_key() or keys[0]

        return ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# Global singleton instance
key_pool = APIKeyPool()
