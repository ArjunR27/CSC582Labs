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
import random
import spacy
from dotenv import load_dotenv
import os
from groq import Groq
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


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
    "Periodic table", "Covalent bond", "DNA",
    "Protein folding", "Mitochondria", "Photosynthesis",
    # Space
    "Large Hadron Collider", "James Webb Space Telescope",
    "Hubble Space Telescope", "Neutron star", "Pulsar",
    "NASA", "SpaceX", "Rocket propulsion", "Black holes", "Exoplanets"
]

TOPIC_DOCS = [(topic, nlp(topic)) for topic in TOPICS]

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
    
    def ask_llm(self, prompt):
        context = prompt
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

                        Specifically your information should come from the following context: {context}
                    """,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response.choices[0].message.content

    def keyword_extraction(self, query):
        doc = nlp(query)
        keyword_chunks = []
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and chunk.root.pos_ != 'PRON':
                keyword_chunks.append(chunk.text.lower())
        return keyword_chunks

    def get_topic(self, query):
        keyword_chunks = self.keyword_extraction(query)
        print(keyword_chunks)
        if not keyword_chunks:
            return random.choice(TOPICS)

        search_text = " ".join(keyword_chunks)
        search_doc = nlp(search_text)
    
        best_topic, best_score = None, 0.0
        for topic, topic_doc in TOPIC_DOCS:
            if not topic_doc.has_vector or topic_doc.vector_norm == 0:
                continue
            score = search_doc.similarity(topic_doc)
            if score > best_score:
                best_score = score
                best_topic = topic
        
        if best_score > 0.3:
            return best_topic
        
        else:
            return random.choice(TOPICS)


    # def get_topic(self, query):
    #     # extract important words from query: 
    #     for topic, topic_doc in TOPIC_DOCS:
    #         print(topic, topic_doc.has_vector, topic_doc.vector_norm)

    #     query_doc = nlp(query)
    #     if not query_doc.has_vector:
    #         return random.choice(TOPICS)
        
    #     best_topic, best_score = None, 0.0
    #     for topic, topic_doc in TOPIC_DOCS:
    #         score = query_doc.similarity(topic_doc)
    #         if score > best_score:
    #             best_score = score
    #             best_topic = topic
        
    #     return best_topic if best_score > 0.3 else random.choice(TOPICS)


    def fetch_wiki_fact(self, topic):
        try:
            summary = wikipedia.summary(topic, sentences=10, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                summary = wikipedia.summary(e.options[0], sentences=10, auto_suggest=False)
            except Exception:
                return None
        except Exception:
            return None
        sentences = [s.strip() for s in summary.split(".") if len(s.strip()) > 30]
        if len(sentences) < 3:
            return None
        start = random.randint(0, max(0, len(sentences) - 3))
        chunk = sentences[start:start + random.randint(3, 4)]
        return ". ".join(chunk) + "."

    def generate_wiki_response(self, text):
        topic = self.extract_topic(text)
        print(topic)
        if topic:
            fact = self.fetch_wiki_fact(topic)
            print(fact)
            # Its taking hella long to generate
            # if fact:
            #     return self.ask_llm(fact)
            return self.say(fact)
        return "You don't know what you're talking to why would I even respond to such low IQ. "
    
    def personality_tick(self):
        return

if __name__ == "__main__":
    sheldon = Sheldon(None, None, None)
    query = "I love racing cars"
    print(query)
    topic = sheldon.get_topic(query)
    print(topic)