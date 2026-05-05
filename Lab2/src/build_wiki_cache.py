"""Generate a local Wikipedia cache for Sheldon TOPICS."""

import json
from pathlib import Path

from archetypes.sheldon import Sheldon, TOPICS

CACHE_PATH = Path(__file__).resolve().parent / "wiki_topic_cache.json"


def main():
    sheldon = Sheldon(conn=None, channel=None, bot=None)

    cache = {}
    total = len(TOPICS)
    print(f"Building Wikipedia cache for {total} topics: ")

    for idx, topic in enumerate(TOPICS, start=1):
        article = sheldon.fetch_wiki_fact(topic)
        if article:
            cache[topic] = article
            print(f"[{idx}/{total}] OK: {topic} -> {article.get('title')}")
        else:
            print(f"[{idx}/{total}] FAIL: {topic}")

    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(cache)} articles to {CACHE_PATH}")


if __name__ == "__main__":
    main()
