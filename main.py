import re
from collections import deque
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Callable

# =====================================================================
# 1. TOKENIZATION & ARCHIVAL DATA STRUCTURES
# =====================================================================


def tokenize(text: str) -> set[str]:
    """Extract clean lowercase alphanumeric tokens ignoring punctuation."""
    return set(re.findall(r"\w+", text.lower()))


@dataclass
class ArchivalRecord:
    rid: str
    text: str
    tags: tuple[str, ...] = ()


class ArchivalStore:
    """Long-term memory store supporting token-based similarity search."""

    def __init__(self) -> None:
        self.records: list[ArchivalRecord] = []
        self._counter: int = 0

    def insert(self, text: str, tags: tuple[str, ...] = ()) -> str:
        self._counter += 1
        rid = f"a{self._counter:03d}"
        record = ArchivalRecord(rid=rid, text=text, tags=tags)
        self.records.append(record)
        return rid

    def search(self, query: str, top_k: int = 3) -> list[ArchivalRecord]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scored: list[tuple[float, ArchivalRecord]] = []
        for rec in self.records:
            r_tokens = tokenize(rec.text)
            union = q_tokens | r_tokens
            score = len(q_tokens & r_tokens) / len(union) if union else 0.0
            if score > 0:
                scored.append((score, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def count(self) -> int:
        return len(self.records)


# =====================================================================
# 2. MAIN WORKING CONTEXT (RAM)
# =====================================================================


@dataclass
class Message:
    role: str
    text: str


@dataclass
class MainContext:
    """Working memory buffer with max capacity eviction and rendering."""

    max_messages: int = 3
    core: dict[str, str] = field(default_factory=dict)
    messages: deque[Message] = field(default_factory=deque)
    evicted: list[Message] = field(default_factory=list)

    def append(self, role: str, text: str) -> None:
        """FIFO sliding window with O(1) deque eviction."""
        self.messages.append(Message(role=role, text=text))
        while len(self.messages) > self.max_messages:
            self.evicted.append(self.messages.popleft())

    def render(self) -> str:
        """Serialize memory into a deterministic prompt context string."""
        parts: list[str] = ["[core]"]
        if not self.core:
            parts.append("  (empty)")
        for key, value in sorted(self.core.items()):
            parts.append(f"  {key}: {value}")

        parts.append("[messages]")
        for msg in self.messages:
            parts.append(f"  {msg.role}: {msg.text}")
        return "\n".join(parts)


# =====================================================================
# 3. MEMORY TOOLS INTERFACE
# =====================================================================


class MemoryTools:
    """Safe state manipulation methods exposed to the agent dispatch layer."""

    def __init__(self, main: MainContext, archival: ArchivalStore) -> None:
        self.main = main
        self.archival = archival

    def core_memory_append(self, section: str, text: str) -> str:
        existing = self.main.core.get(section, "")
        updated = f"{existing}, {text}".strip(", ") if existing else text
        self.main.core[section] = updated
        return f"core[{section}] appended: {len(text)} chars"

    def core_memory_replace(self, section: str, old_text: str, new_text: str) -> str:
        if section not in self.main.core:
            return f"error: section {section!r} not found"
        val = self.main.core[section]
        if old_text not in val:
            return f"error: text {old_text!r} not found in core[{section}]"
        self.main.core[section] = val.replace(old_text, new_text)
        return f"core[{section}] replaced"

    def archival_memory_insert(self, text: str, tags: tuple[str, ...] = ()) -> str:
        tags_tuple = tuple(tags) if isinstance(tags, (list, tuple)) else ()
        rid = self.archival.insert(text, tags=tags_tuple)
        return f"stored {rid} ({self.archival.count()} records)"

    def archival_memory_search(self, query: str, top_k: int = 3) -> str:
        hits = self.archival.search(query, top_k=int(top_k))
        if not hits:
            return "no matches"
        return "\n".join(f"  {h.rid}: {h.text}" for h in hits)

    def conversation_search(self, query: str) -> str:
        """Search messages backwards from newest to oldest without array allocation."""
        q = query.lower()
        for msg in chain(reversed(self.main.messages), reversed(self.main.evicted)):
            if q in msg.text.lower():
                return f"found ({msg.role}): {msg.text}"
        return "no matches"


# =====================================================================
# 4. DISPATCHER & AGENT RUNTIME
# =====================================================================


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


class ToolDispatcher:
    """Explicit tool registry preventing arbitrary dynamic execution."""

    def __init__(self, tools: MemoryTools) -> None:
        self._registry: dict[str, Callable[..., str]] = {
            "core_memory_append": tools.core_memory_append,
            "core_memory_replace": tools.core_memory_replace,
            "archival_memory_insert": tools.archival_memory_insert,
            "archival_memory_search": tools.archival_memory_search,
            "conversation_search": tools.conversation_search,
        }

    def dispatch(self, call: ToolCall) -> str:
        handler = self._registry.get(call.name)
        if handler is None:
            return f"error: unknown tool {call.name!r}"
        try:
            return handler(**call.args)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"


def run_scripted_agent(dispatcher: ToolDispatcher, script: list[ToolCall]) -> list[str]:
    return [dispatcher.dispatch(call) for call in script]


# =====================================================================
# 5. EXECUTION SUITE
# =====================================================================

if __name__ == "__main__":
    main_ctx = MainContext(max_messages=3)
    archival = ArchivalStore()
    tools = MemoryTools(main_ctx, archival)
    dispatcher = ToolDispatcher(tools)

    # 1. Initial Interaction Sequence
    main_ctx.append(
        "user", "hey I'm marcus, staff engineer working on distributed storage"
    )
    main_ctx.append(
        "assistant", "welcome marcus! what storage bottlenecks are you benchmarking?"
    )
    main_ctx.append(
        "user", "we are testing postgres lsm trees vs rocksdb for write-heavy workloads"
    )
    main_ctx.append(
        "assistant",
        "rocksdb handles write-heavy workloads better due to lsm tree structure",
    )

    # 2. Scripted Tool Invocations
    script = [
        ToolCall(
            "core_memory_append",
            {"section": "persona", "text": "agent is a distributed systems specialist"},
        ),
        ToolCall(
            "core_memory_append",
            {"section": "user", "text": "name=marcus, role=staff storage engineer"},
        ),
        ToolCall(
            "core_memory_replace",
            {
                "section": "user",
                "old_text": "role=staff storage engineer",
                "new_text": "role=principal storage engineer",
            },
        ),
        ToolCall(
            "archival_memory_insert",
            {
                "text": "rocksdb uses dynamic levelling compaction to reduce write amplification",
                "tags": ("database", "rocksdb", "benchmarks"),
            },
        ),
        ToolCall(
            "archival_memory_insert",
            {
                "text": "postgres B-tree indexes suffer page splits under random write traffic",
                "tags": ("database", "postgres"),
            },
        ),
        ToolCall(
            "archival_memory_insert",
            {
                "text": "raft consensus log replication needs disk fsync under 5ms latency",
                "tags": ("raft", "consensus"),
            },
        ),
    ]

    observations = run_scripted_agent(dispatcher, script)

    print("\n--- TOOL TRACE (MEMORY WRITES & MUTATIONS) ---")
    for call, obs in zip(script, observations):
        print(f"  {call.name}({call.args}) -> {obs}")

    # 3. Filling Main Context to Trigger Eviction
    print("\n--- FILLING MAIN CONTEXT (EVICTION TRIGGER) ---")
    main_ctx.append("user", "what was that index bottleneck you mentioned earlier?")
    main_ctx.append(
        "assistant", "let me search our archival memory for index write issues."
    )

    print(
        f"\nmain context ({len(main_ctx.messages)} active messages, {len(main_ctx.evicted)} evicted messages)"
    )
    print(main_ctx.render())

    # 4. Memory Retrieval Testing
    print("\n--- TEST 1: Archival Search (Jaccard Match) ---")
    print("Query: 'postgres B-tree index'")
    hit_1 = tools.archival_memory_search("postgres B-tree index", top_k=1)
    print(hit_1)

    print("\n--- TEST 2: Conversation Search (Evicted History Match) ---")
    print("Query: 'lsm trees'")
    hit_2 = tools.conversation_search("lsm trees")
    print(hit_2)

    print("\n--- TEST 3: Conversation Search (No Matches Fallback) ---")
    print("Query: 'redis cache'")
    hit_3 = tools.conversation_search("redis cache")
    print(hit_3)
