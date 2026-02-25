"""
Command-line interface for cache-for-clankers.

Sub-commands
------------
store   – Store a piece of text in memory.
retrieve – Retrieve the most relevant memories for a query.
list    – List all stored memories.
delete  – Delete a memory by its ID.
count   – Print the number of stored memories.
"""

from __future__ import annotations

import argparse
import json
import sys

from .memory import MemoryManager


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cache-for-clankers",
        description="Locally hosted semantic memory for LLM sessions.",
    )
    parser.add_argument(
        "--db",
        default="./chroma_db",
        metavar="PATH",
        help="Path to the ChromaDB persistent store (default: ./chroma_db).",
    )
    parser.add_argument(
        "--collection",
        default="memories",
        metavar="NAME",
        help="ChromaDB collection name (default: memories).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # store
    p_store = sub.add_parser("store", help="Store text in memory.")
    p_store.add_argument("text", nargs="?", help="Text to store (reads stdin if omitted).")
    p_store.add_argument("--session", default=None, help="Optional session identifier.")
    p_store.add_argument(
        "--no-chunk",
        action="store_true",
        help="Disable automatic chunking of long texts.",
    )

    # retrieve
    p_retrieve = sub.add_parser("retrieve", help="Retrieve relevant memories.")
    p_retrieve.add_argument("query", help="Natural-language query.")
    p_retrieve.add_argument(
        "-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to return (default: 5).",
    )
    p_retrieve.add_argument(
        "--min-importance",
        type=float,
        default=0.0,
        metavar="SCORE",
        help="Minimum importance score filter (default: 0.0).",
    )
    p_retrieve.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output results as JSON.",
    )

    # list
    p_list = sub.add_parser("list", help="List stored memories.")
    p_list.add_argument(
        "--limit",
        type=int,
        default=100,
        metavar="N",
        help="Maximum number of memories to show (default: 100).",
    )
    p_list.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON.")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a memory by ID.")
    p_delete.add_argument("id", help="Memory ID to delete.")

    # count
    sub.add_parser("count", help="Print the number of stored memories.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    manager = MemoryManager(db_path=args.db, collection_name=args.collection)

    if args.command == "store":
        text = args.text
        if text is None:
            text = sys.stdin.read()
        if not text.strip():
            print("Error: no text provided.", file=sys.stderr)
            return 1
        ids = manager.store(
            text,
            session_id=args.session,
            auto_chunk=not args.no_chunk,
        )
        print(f"Stored {len(ids)} memory chunk(s): {', '.join(ids)}")

    elif args.command == "retrieve":
        results = manager.retrieve(args.query, n_results=args.n, min_importance=args.min_importance)
        if not results:
            print("No memories found.")
            return 0
        if args.as_json:
            print(json.dumps(results, indent=2))
        else:
            for i, r in enumerate(results, 1):
                print(f"[{i}] (similarity={r['similarity']:.3f}, "
                      f"importance={r['metadata'].get('importance', 0):.3f})")
                print(f"    {r['content'][:200]}")
                print(f"    id={r['id']}")
                print()

    elif args.command == "list":
        memories = manager.list_all(limit=args.limit)
        if not memories:
            print("No memories stored.")
            return 0
        if args.as_json:
            print(json.dumps(memories, indent=2))
        else:
            for m in memories:
                ts = m["metadata"].get("timestamp", "")
                imp = m["metadata"].get("importance", 0)
                print(f"id={m['id']} ts={ts:.0f} importance={imp:.3f}" if ts else
                      f"id={m['id']} importance={imp:.3f}")
                print(f"    {m['content'][:120]}")
                print()

    elif args.command == "delete":
        manager.delete(args.id)
        print(f"Deleted memory {args.id}.")

    elif args.command == "count":
        print(manager.count())

    return 0


if __name__ == "__main__":
    sys.exit(main())
