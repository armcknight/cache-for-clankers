# cache-for-clankers

A long-term memory solution for LLMs.

`cache-for-clankers` provides a locally hosted vector database — backed by
[ChromaDB](https://www.trychroma.com/) — with an intelligent logic layer that
lets Claude (and other LLMs) persist important context across sessions and
compactions.

---

## Installing as a Claude plugin (MCP)

`cache-for-clankers` ships an [MCP](https://modelcontextprotocol.io/) server
so Claude Desktop can call it as a set of built-in tools.

### 1 — Install the package

```bash
pip install -e .
```

### 2 — Add it to Claude Desktop's config

Open (or create) `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) and add the
server under `"mcpServers"`:

```json
{
  "mcpServers": {
    "cache-for-clankers": {
      "command": "cache-for-clankers-mcp"
    }
  }
}
```

Restart Claude Desktop. You will now see five new tools available inside Claude:
`store_memory`, `retrieve_memories`, `list_memories`, `delete_memory`, and
`count_memories`.

### Optional environment variables

| Variable | Default | Description |
|---|---|---|
| `CACHE_FOR_CLANKERS_DB_PATH` | `~/.cache/cache-for-clankers` | Path to the ChromaDB persistent store |
| `CACHE_FOR_CLANKERS_COLLECTION` | `memories` | ChromaDB collection name |
| `CACHE_FOR_CLANKERS_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers embedding model |

Example with custom path:

```json
{
  "mcpServers": {
    "cache-for-clankers": {
      "command": "cache-for-clankers-mcp",
      "env": {
        "CACHE_FOR_CLANKERS_DB_PATH": "/path/to/my/memory"
      }
    }
  }
}
```

---

## Why the intelligence layer matters

ChromaDB is an excellent general-purpose vector store, but it provides only
**raw storage and similarity search**. By itself it will:

- Store duplicate or near-duplicate content without merging it, wasting space
  and polluting results with redundant entries.
- Return results ranked purely by embedding similarity, with no concept of
  *how dense* or *how informative* the stored text is.
- Store long context blobs whole, making retrieval return a wall of text
  rather than the precise relevant fragment.

The intelligence layer adds three things ChromaDB does not:

| Layer | What it does |
|---|---|
| **Semantic chunking** | Splits long texts on paragraph/sentence boundaries *before* embedding so that retrieval returns precise fragments, not walls of text |
| **Importance scoring** | Assigns a 0-1 information-density score to each chunk (vocabulary richness + structure markers + numeric content) and uses it to *re-rank* retrieval results |
| **Near-duplicate deduplication** | Before every write, checks cosine similarity against existing entries; if >= 0.92 it merges rather than inserts, keeping the store compact |

---

## MCP tools reference

| Tool | Description |
|---|---|
| `store_memory(content, session_id?)` | Save an important piece of context. Chunks, deduplicates, and scores automatically. |
| `retrieve_memories(query, n_results?, min_importance?)` | Semantic search re-ranked by similarity + importance. |
| `list_memories(limit?)` | Browse all stored memories. |
| `delete_memory(memory_id)` | Remove a specific memory by ID. |
| `count_memories()` | How many memories are stored right now. |

---

## Python API

```python
from cache_for_clankers import MemoryManager

memory = MemoryManager(db_path="./my_memory")

# End of session -- store context fragments
ids = memory.store(
    "The user is Alice. She is a senior Python developer who prefers "
    "type-annotated code and dislikes JavaScript.",
    session_id="session-001",
)

# Start of next session -- retrieve relevant context
results = memory.retrieve("What does the user prefer?", n_results=3)
for r in results:
    print(f"[{r['similarity']:.2f}] {r['content']}")
```

---

## CLI

```bash
# Store from a string or piped stdin
cache-for-clankers store "Alice prefers typed Python."
echo "She dislikes JavaScript." | cache-for-clankers store

# Retrieve relevant memories
cache-for-clankers retrieve "What language does Alice prefer?" -n 5
cache-for-clankers retrieve --json "Alice's preferences"

# Manage
cache-for-clankers list
cache-for-clankers delete <id>
cache-for-clankers count
```

---

## Architecture

```
+--------------------------------------------------+
|          MCP Server (Claude plugin)              |
|  store_memory  retrieve_memories  list_memories  |
|  delete_memory  count_memories                   |
+---------------------+----------------------------+
                      |
+---------------------v----------------------------+
|               MemoryManager (API)                |
|  store()  retrieve()  delete()  list_all()       |
+---------------------+----------------------------+
                      |
+---------------------v----------------------------+
|          Intelligence Layer                      |
|  chunk_text()  compute_importance()              |
|  deduplicate_content()                           |
+---------------------+----------------------------+
                      |
+---------------------v----------------------------+
|          VectorStore (ChromaDB wrapper)          |
|  add()  update()  query()  delete()  get_all()   |
+---------------------+----------------------------+
                      |
+---------------------v----------------------------+
|     ChromaDB (persistent, cosine-similarity)     |
|     sentence-transformers  (all-MiniLM-L6-v2)    |
+--------------------------------------------------+
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
