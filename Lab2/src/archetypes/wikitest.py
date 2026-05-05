import wikipedia
import re

# Search and get top results
results = wikipedia.search("transformer neural network", results=5)
print(results)  # ['Transformer (deep learning)', ...]


# Grab a page
page = wikipedia.page(results[0])
clean = re.sub(r'==+[^=]+=+\n?', '', page.content)
print(clean)