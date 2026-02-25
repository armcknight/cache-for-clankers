# cache-for-clankers

A long-term memory solution for LLMs.

`cache-for-clankers` provides a locally hosted vector database — backed by
[ChromaDB](https://www.trychroma.com/) — with an intelligent logic layer that
lets Claude (and other LLMs) persist important context across sessions and
compactions.

---

## Features

| Capability | Description |
|---|---|
| **Local storage** | ChromaDB runs embedded in the same process — no external server required. |
| **Semantic embeddings** | Text is embedded with [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2` by default). |
| **Smart chunking** | Long texts are split into semantically coherent paragraph/sentence chunks before storage. |
| **Deduplication** | Near-duplicate content (cosine similarity ≥ 0.92) is merged rather than stored twice, keeping the database lean. |
| **Importance scoring** | Each chunk receives a heuristic importance score (vocabulary richness, structure, facts) used to re-rank retrieval results. |
| **Contextual retrieval** | Results are ranked by a weighted blend of semantic similarity (70 %) and importance score (30 %). |

---

## Installation

```bash
pip install -e .
```

> **Dependencies:** `chromadb>=0.5`, `sentence-transformers>=3.0`
> (both are installed automatically).

---

## Quick start — Python API

```python
from cache_for_clankers import MemoryManager

# Create (or reopen) a persistent memory store
memory = MemoryManager(db_path="./my_memory")

# --- At the end of a Claude session ---
ids = memory.store(
    "The user is Alice. She is a senior Python developer who prefers "
    "type-annotated code and dislikes JavaScript.",
    session_id="session-001",
)

# --- At the start of the next session ---
results = memory.retrieve("What does the user prefer?", n_results=3)
for r in results:
    print(f"[{r['similarity']:.2f}] {r['content']}")
```

### Additional API methods

```python
# List everything stored
for m in memory.list_all():
    print(m["id"], m["content"][:80])

# Delete a specific memory
memory.delete(ids[0])

# Count stored memories
print(memory.count())
```

---

## Quick start — CLI

After installation the `cache-for-clankers` command is available:

```bash
# Store a memory (from a string or piped stdin)
cache-for-clankers store "Alice prefers Python with type annotations."
echo "She dislikes JavaScript." | cache-for-clankers store

# Retrieve relevant memories
cache-for-clankers retrieve "What language does Alice prefer?" -n 5

# Retrieve as JSON (useful for scripting)
cache-for-clankers retrieve --json "Alice's preferences"

# List all stored memories
cache-for-clankers list
cache-for-clankers list --json --limit 20

# Delete a memory by ID
cache-for-clankers delete <id>

# Count memories
cache-for-clankers count
```

Use `--db` and `--collection` to change where data is stored:

```bash
cache-for-clankers --db /path/to/db --collection project-x store "..."
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│               MemoryManager (API)               │
│  store()  retrieve()  delete()  list_all()      │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│          Intelligence Layer                     │
│  chunk_text()  compute_importance()             │
│  deduplicate_content()                          │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│          VectorStore (ChromaDB wrapper)         │
│  add()  update()  query()  delete()  get_all()  │
└────────────┬────────────────────────────────────┘
             │
┌────────────▼────────────────────────────────────┐
│     ChromaDB (persistent, cosine-similarity)    │
│     sentence-transformers  (all-MiniLM-L6-v2)   │
└─────────────────────────────────────────────────┘
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use an in-memory ChromaDB instance with a deterministic fake embedding
function — no model downloads needed.
