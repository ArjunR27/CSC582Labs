"""
facts “nerd”, know-it-all, arrogant
- volunteers random information about random geeky subjects
- wants to be the center of attention, particularly hates two other users talking to each other that ignore him
- snide put downs on intelligence

Data Source
- Wikipedia, but only certain subjects
- a geeky/tech ontology
"""
import wikipedia
import spacy
from dotenv import load_dotenv
import os
from groq import Groq


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
nlp = spacy.load("en_core_web_md")

TOPICS =  [
    # Physics
    "Quantum mechanics", "String theory", "Supersymmetry", "Dark matter",
    "General relativity", "Special relativity", "Schrödinger equation",
    "Higgs boson", "Standard Model", "Thermodynamics", "Entropy",
    "Electromagnetic spectrum", "Photon", "Neutrino", "Black hole",
    # Computer Science
    "Alan Turing", "Turing machine", "P versus NP problem",
    "Computational complexity theory", "Von Neumann architecture",
    "Quantum computing", "Algorithm", "Binary number",
    # Math
    "Prime number", "Riemann hypothesis", "Euler's identity",
    "Game theory", "Topology", "Fermat's Last Theorem",
    # Chemistry / Biology
    "Periodic table", "Covalent bond", "DNA", "CRISPR",
    "Protein folding", "Mitochondria", "Photosynthesis",
    # Space
    "Large Hadron Collider", "James Webb Space Telescope",
    "Hubble Space Telescope", "Neutron star", "Pulsar",
    "NASA", "SpaceX", "Rocket propulsion", "Black holes", "Exoplanets"
]

TOPIC_DOCS = [(topic, nlp.make_doc(topic)) for topic in TOPICS]

class Sheldon():
    def __init__(self, conn, channel, bot):
        self.name = 'sheldon'
        self.conn = conn
        self.channel = channel
        self.bot = bot
    
    def get_name(self):
        return self.name
    
    def say(self, msg):
        max_len = 400  # leaves room for PRIVMSG framing overhead
        for i in range(0, len(msg), max_len):
            self.conn.privmsg(self.channel, msg[i:i + max_len])
    
    def ask_llm(self, context, query):
        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=[
                {
                    "role": "system",
                    "content": f"""
                        You are a personality named Sheldon. You are a facts nerd, know it all, and arrogant. 
                        - You volunteer random information about random geeky subjects
                        - You want to be the center of attention and particularly hates two other users talking to each other that ignore him
                        - Snide put downs on intelligence
                        - Keep your response to one short paragraph

                        Answer using only the following Wikipedia context:
                        {context}
                    """,
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        return response.choices[0].message.content

    def extract_topic(self, text):
        text_lower = text.lower()

        # Direct keyword match first — far more reliable than vector similarity
        # for queries like "what do you know about prime numbers?"
        for topic, _ in TOPIC_DOCS:
            topic_lower = topic.lower()
            if topic_lower in text_lower:
                return topic
            # All significant words (4+ chars) from the topic appear in the text
            words = [w for w in topic_lower.split() if len(w) > 3]
            if words and all(w in text_lower for w in words):
                return topic

        # Fall back to vector similarity for paraphrased/indirect mentions
        user_doc = nlp.make_doc(text)
        if not user_doc.has_vector:
            return None
        best_topic = None
        best_score = 0.0
        for topic, topic_doc in TOPIC_DOCS:
            if not topic_doc.has_vector:
                continue
            score = user_doc.similarity(topic_doc)
            if score > best_score:
                best_topic = topic
                best_score = score

        if best_score > 0.3:
            return best_topic
        return None

    def fetch_wiki_fact(self, topic):
        try:
            results = wikipedia.search(topic)
            if not results:
                return None

            page = wikipedia.page(results[0], auto_suggest=False)
            return {
                "title": page.title,
                "url": page.url,
                "summary": wikipedia.summary(page.title, sentences=5, auto_suggest=False),
                "content": page.content,
            }
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return {
                    "title": page.title,
                    "url": page.url,
                    "summary": wikipedia.summary(page.title, sentences=5, auto_suggest=False),
                    "content": page.content,
                }
            except Exception:
                return None
        except Exception:
            return None

    def wiki_extract_from_query(self, query, wiki_text, chunk_size=3, top_k=2):
        if not wiki_text or not query:
            return None

        doc = nlp(wiki_text)

        # create sentences
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        if not sentences:
            return None

        # create chunks, groups of 3 sentences
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size]).strip()
            if len(chunk) >= 80:
                chunks.append(chunk)

        if not chunks:
            return None

        query_doc = nlp(query)
        scored_chunks = []

        # keep only main query terms

        query_terms = {token.lemma_.lower() for token in query_doc if token.is_alpha and not token.is_stop}

        for chunk in chunks:
            chunk_doc = nlp(chunk)

            # sim score between query doc and chunk doc
            sim_score = query_doc.similarity(chunk_doc) if query_doc.has_vector and chunk_doc.has_vector else 0.0

            final_score = sim_score

            scored_chunks.append((final_score, chunk))

        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        best_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
        top_score = scored_chunks[0][0] if scored_chunks else 0.0
        return {
            "context": "\n\n".join(best_chunks),
            "score": top_score,
        }

    def generate_wiki_response(self, text):
        min_extract_score = 0.30
        topic = self.extract_topic(text)
        if topic:
            article = self.fetch_wiki_fact(topic)
        else:
            article = None

        if article:
            extracted = self.wiki_extract_from_query(text, article.get("content", ""))
            context = None

            if extracted and extracted["score"] >= min_extract_score:
                context = extracted["context"]
            if not context:
                context = article.get("summary", "")

            llm_response = self.ask_llm(context, text)
            if llm_response:
                return self.say(llm_response)
            response = f"{article['title']}: {context}"
            return self.say(response)
        return "You don't know what you're talking to why would I even respond to such low IQ. "
    
    def personality_tick(self):
        return