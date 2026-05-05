"""Simple local test for Sheldon Wikipedia retrieval + query extraction."""

from archetypes.sheldon import Sheldon


def run_test():
    sheldon = Sheldon(conn=None, channel=None, bot=None)

    topic = "Black hole"
    query = "who discovered black holes"

    article = sheldon.fetch_wiki_fact(topic)

    if not article:
        print("Failed to fetch Wikipedia article.")
        return

    extracted = sheldon.wiki_extract_from_query(query, article.get("content", ""))
    extracted_text = extracted["context"] if extracted else "No relevant context found."
    extracted_score = extracted["score"] if extracted else 0.0
    llm_answer = sheldon.ask_llm(extracted_text, query)

    print(f"Topic: {topic}")
    print(f"Query: {query}")
    print(f"Title: {article.get('title')}")
    print(f"URL: {article.get('url')}")
    print(f"Extract Score: {extracted_score:.4f}")
    print("\nTop extracted context:\n")
    print(extracted_text)
    print("\nLLM formatted response:\n")
    print(llm_answer)


if __name__ == "__main__":
    run_test()
