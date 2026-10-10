# Minimal MemGPT in Python 

A small, dependency-free implementation of a MemGPT-style memory engine for AI agents.

The project explores how an agent can manage limited working memory, preserve older conversation messages, and retrieve relevant information when needed.

## Features

- **Working memory:** Keeps recent messages in a bounded sliding window.
- **Core memory:** Stores persistent facts in named sections that can be appended to or updated.
- **Archival memory:** Stores long-term facts and notes.
- **Keyword search:** Uses Jaccard similarity to find relevant archived records.
- **Conversation search:** Searches active and evicted conversation messages.
- **Tool dispatcher:** Exposes memory operations through an explicit tool registry.

## Getting Started

Requires Python 3.10+.

Clone the repository, navigate to the project directory, and run:

```bash
python3 main.py
```

No third-party dependencies are required.

## How It Works

1. **Keep recent context** — `MainContext` stores a limited number of recent messages.
2. **Evict older messages** — When the limit is reached, the oldest messages move to the evicted history.
3. **Store important facts** — `ArchivalStore` holds facts that may be needed later.
4. **Retrieve information** — Memory tools search archived records or conversation history.
5. **Dispatch tool calls** — `ToolDispatcher` routes named tool calls to the appropriate memory operation.

## Memory Layers

| Layer           | Purpose                                        |
| --------------- | ---------------------------------------------- |
| Core memory     | Stores facts in named sections                 |
| Working memory  | Holds recent conversation messages             |
| Archival memory | Stores long-term facts and notes               |
| Evicted history | Preserves messages removed from working memory |

## Example

The included script demonstrates:

- Updating core memory
- Archiving technical facts
- Evicting older conversation messages
- Searching archival records
- Retrieving information from earlier conversation history

## Limitations

This is an educational prototype, not a complete MemGPT implementation.

- Memory is stored in process and is lost when the program exits.
- Archival search uses lexical similarity, not embeddings or semantic search.
- Memory insertion is explicit; the engine does not automatically decide which facts to remember.
- The included agent runtime uses scripted tool calls rather than an LLM-driven agent loop.

