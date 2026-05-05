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
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import random
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json
import os

BERT_TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
BERT_MODEL = BertModel.from_pretrained("bert-base-uncased")
BERT_MODEL.eval()


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
        self.register_activity()
        msg = msg.replace('\r', '').replace('\n', ' ')
        max_len = 400  # leaves room for PRIVMSG framing overhead
        for i in range(0, len(msg), max_len):
            self.conn.privmsg(self.channel, msg[i:i + max_len])

    def register_activity(self):
        return
    
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

    def keyword_extraction(self, query):
        doc = nlp(query)
        keyword_chunks = []
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and chunk.root.pos_ != 'PRON':
                keyword_chunks.append(chunk.text.lower())
        return keyword_chunks

    def get_topic(self, query):
        keyword_chunks = self.keyword_extraction(query)
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
    
    def get_cls_vector(self, wiki_text):
        inputs = BERT_TOKENIZER(
            wiki_text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        with torch.no_grad():
            outputs = BERT_MODEL(**inputs)
        
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    def _is_clean_sentence(self, text):
        """Return False for sentences with newlines, LaTeX markup, or other noise."""
        if '\n' in text:
            return False
        if '{\\displaystyle' in text or '\\displaystyle' in text:
            return False
        if text.count('{') + text.count('}') > 4:
            return False
        return True

    def fact_extractor(self, wiki_text, top_n=3):
        print("Extracting Fact!")
        if not wiki_text:
            return None

        doc = nlp(wiki_text)
        sentences = []
        for sent in doc.sents:
            text = sent.text.strip()
            if len(text) > 10 and self._is_clean_sentence(text):
                sentences.append(text)
        
        if not sentences:
            return None
    
        max_start = max(0, len(sentences) - 15)
        start = random.randint(0, max_start)
        chunk = sentences[start:start + 15]

        scored = []
        for sent in chunk:
            inputs = BERT_TOKENIZER(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = BERT_MODEL(**inputs)
            cls_vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            score = np.linalg.norm(cls_vec)
            scored.append((score, sent))

        scored.sort(reverse=True)
        top_pool = scored[:max(top_n, len(scored) // 2)]
        chosen = random.sample(top_pool, min(top_n, len(top_pool)))
        return " ".join([s for _, s in chosen])



    def wiki_extract_from_query(self, query, wiki_text, chunk_size=3, top_k=2):
        if not wiki_text or not query:
            return None

        doc = nlp(wiki_text)

        # create sentences, skipping noisy ones (newlines, LaTeX markup)
        sentences = [
            sent.text.strip() for sent in doc.sents
            if sent.text.strip() and self._is_clean_sentence(sent.text.strip())
        ]
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

    def generate_wiki_response(self, text, author=None):
        self.register_activity()
        min_extract_score = 0.30
        topic = self.get_topic(text)
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
                if author:
                    llm_response = f"{author}: {llm_response}"
                return self.say(llm_response)
            response = f"{article['title']}: {context}"
            if author:
                response = f"{author}: {response}"
            return self.say(response)
        return "You don't know what you're talking to why would I even respond to such low IQ. "
    
    def personality_tick(self):
        # ~15% chance per tick: reach out and greet a random channel member.
        if random.random() < 0.50:
            self.bot.initiate_greeting_to_random_user()
            return

        json_cache = os.path.join(os.path.dirname(__file__), '..', 'wiki_topic_cache.json')
        with open(json_cache, 'r') as f:
            cache = json.load(f)

        topic = random.choice(list(cache.keys()))
        wiki_text = cache[topic].get('content', '')

        facts = self.fact_extractor(wiki_text)
        if facts:
            self.say(f"Here are some fun facts: {facts}")
