import wikipedia

def get_wikipedia_info(topic, sentences=5):
    try:
        results = wikipedia.search(topic)

        return results[0:5]

        if not results:
            return None

        page = wikipedia.page(results[0], auto_suggest=False)

        return {
            "title": page.title,
            "url": page.url,
            "summary": wikipedia.summary(page.title, sentences=sentences),
            "content": page.content
        }

    except wikipedia.DisambiguationError as e:
        return {
            "error": "Topic is ambiguous",
            "options": e.options[:10]
        }

    except wikipedia.PageError:
        return {
            "error": "Page not found"
        }

topic = "python"
info = get_wikipedia_info(topic)

print(info)

# print(info["title"])
# print(info["summary"])
# print(info.keys())
# print(info["content"])