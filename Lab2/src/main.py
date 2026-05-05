from irc.bot import SingleServerIRCBot
import sys
from archetypes import sheldon
import random
from collections import deque, defaultdict
import time
import re
import os
import json


# CHANNEL = "#CSC582"
CHANNEL = "#CSC582"
BOT_DESCRIPTION = (
    "Hello, I am Sheldon. A theoretical physicist at Caltech with a B.S, M.S, and Ph.D from East Texas Tech + a Ph.D from Caltech all before I turned 16. " \
    "If you have a question about computer science, math, or physics, please do ask as I would be glad to elevate your mind although it would never reach an IQ as high as mine." \
)

class PersonalityBot(SingleServerIRCBot):
    SERVER = 'irc.libera.chat'
    PORT = 6667
    ALLOWED_PERSONALITIES = set(['sheldon', 'normal'])

    def __init__(self, channel, nickname):
        super().__init__([(self.SERVER, self.PORT)], nickname, nickname)
        self.channel = channel
        self.nickname = nickname
        self.knowledge = {}
        self.current_personality = None
        self.PERSONALITIES = {}
        self.channel_history = deque(maxlen=300)
        self.user_histories = defaultdict(lambda: deque(maxlen=50))
        self.last_seen = {}
    
    def on_welcome(self, conn, event):
        self.conn = conn
        conn.join(self.channel)
        self.current_personality = sheldon.Sheldon(conn, self.channel, self)
        self._schedule_tick()
    
    # Schedules the personalities own 'tick' every 3-5 seconds
    # will change depending on the current personality
    def _schedule_tick(self):
        # can change the delay
        delay = random.uniform(5, 10)
        self.reactor.scheduler.execute_after(delay, self._personality_tick)

    def _personality_tick(self):
        interjected = self.check_and_interject_bot_to_bot()
        if not interjected and self.current_personality:
            self.current_personality.personality_tick()
        self._schedule_tick()

    def check_and_interject_bot_to_bot(self):
        # Scan a short recent window for two users addressing each other.
        recent = self.get_recent_channel_messages(20)
        pair_counts = defaultdict(int)

        for msg in reversed(recent):
            author = msg.get("author")
            recipient = msg.get("recipient")

            if not author or not recipient:
                continue
            if author == self.nickname or recipient == self.nickname:
                continue
            pair_key = tuple(sorted((author.lower(), recipient.lower())))
            pair_counts[pair_key] += 1

        active_pairs = [pair for pair, count in pair_counts.items() if count >= 2]
        if not active_pairs:
            return False

        chosen_pair = random.choice(active_pairs)
        user = random.choice(list(chosen_pair))
        recipient = chosen_pair[0] if chosen_pair[1] == user else chosen_pair[1]
        fact = self.random_fact()
        self.conn.privmsg(self.channel, f"Hey {user}, stop talking to {recipient}. {fact}")
        if self.current_personality and self.current_personality.get_name() == 'sheldon':
            self.current_personality.register_activity()

        # Clear stored messages for this pair so a new interjection only happens
        # after they start a fresh exchange.
        pair_set = set(chosen_pair)
        filtered_history = deque(maxlen=self.channel_history.maxlen)
        for msg in self.channel_history:
            author = (msg.get("author") or "").lower()
            msg_recipient = (msg.get("recipient") or "").lower()
            if author in pair_set and msg_recipient in pair_set:
                continue
            filtered_history.append(msg)
        self.channel_history = filtered_history
        return True

    def random_fact(self):
        if self.current_personality and self.current_personality.get_name() == 'sheldon':
            json_cache = os.path.join(os.path.dirname(__file__), 'wiki_topic_cache.json')
            try:
                with open(json_cache, 'r') as f:
                    cache = json.load(f)
                if cache:
                    topic = random.choice(list(cache.keys()))
                    wiki_text = cache[topic].get('content', '')
                    if wiki_text:
                        fact = self.current_personality.fact_extractor(wiki_text)
                        if fact:
                            return fact
            except Exception:
                pass
        return "did you know 2 + 2 = 4?"
    
    def on_join(self, conn, event):
        if event.source.nick == conn.get_nickname():
            print("hello")
            conn.privmsg(self.channel, f"I have joined the channel!")
            return

    def on_part(self, conn, event):
        nick = event.source.nick
        if nick == conn.get_nickname():
            return

    def on_quit(self, conn, event):
        nick = event.source.nick
        if nick == conn.get_nickname():
            return
    
    # That chatbot is expected to respond to any utterance in the channel that begins with its name followed by an immediate colon (:) symbol.
    def parse_privmsg(self, conn, text, botnick, channel):
        botnick = conn.get_nickname()
        bot_prefix = f"{botnick}:"
        if text.lower().startswith(bot_prefix.lower()):
            command = text[len(bot_prefix):].strip()
            return command
        return None

    # The chatbot must kill itself when command “die” is given to it (preceded by its name followed by colon).
    def handle_die(self, conn, channel, author):
        conn.privmsg(channel, f"{author}: really? OK, fine.")
        self.die(msg="See you later!")
        sys.exit(0)

    def handle_bot_description(self, conn, channel, author):
        conn.privmsg(channel, f"Hello {author}: {BOT_DESCRIPTION}")
        
    def handle_forget(self, conn, channel, author):
        self.knowledge = {}
        self.channel_history = deque(maxlen=300)
        self.user_histories = defaultdict(lambda: deque(maxlen=50))
        self.last_seen = {}
        conn.privmsg(channel, f"{author}: forgetting everything")

    def handle_users(self, conn, channel, author):
        channels = self.channels
        csc_channel = channels[channel]
        users = csc_channel.users()
        user_list = [user for user in users if user != self.nickname]
        if user_list:
            conn.privmsg(channel, f"{author}: Users: " + ", ".join(user_list))
        else:
            conn.privmsg(channel, f"{author}: No users found.")
    
    def handle_hello(self, conn, channel, author):
        conn.privmsg(channel, f"Hello {author}, please stop bothering me. I'mm currently trying to solve the EPR Paradox.")

    def get_recent_channel_messages(self, n=20):
        return list(self.channel_history)[-n:]

    def get_recent_user_messages(self, nick, n=10):
        return list(self.user_histories[nick])[-n:]

    def on_pubmsg(self, conn, event):
        if not event.arguments:
            return
        
        text = event.arguments[0]
        author = event.source.nick
        
        if author == conn.get_nickname():
            return
        
        # add to chat history
        recipient = None
        match = re.match(r"^\s*([^:\s]+)\s*:\s*", text)
        if match:
            recipient = match.group(1)
        addressed_to_bot = bool(recipient and recipient.lower() == conn.get_nickname().lower())

        msg = {
            "ts": time.time(),
            "author": author,
            "text": text,
            "recipient": recipient,
            "addressed_to_bot": addressed_to_bot
        }

        self.channel_history.append(msg)
        self.user_histories[author].append(msg)
        self.last_seen[author] = msg["ts"]
        
        # Commands take priority - check if this is a command first
        command_text = self.parse_privmsg(conn, text, self.nickname, self.channel)
        if command_text:
            if self.current_personality and self.current_personality.get_name() == 'sheldon':
                self.current_personality.register_activity()
            command_lower = command_text.lower().strip()
            normalized_command = re.sub(r"[^a-z0-9\s]", "", command_lower)

            if normalized_command in {"what do you do", "so what do you do"}:
                self.handle_bot_description(conn, self.channel, author)
                return
            
            # Get first word for single-word commands
            parts = command_text.split(None, 1)
            command_name = parts[0].lower()
            command_query = parts[1] if len(parts) > 1 else ""

            print(f"Handling command: {command_name}")
            
            BASE_COMMANDS = {
                # The chatbot must kill itself when command “die” is given to it (preceded by its name followed by colon).
                "die": self.handle_die,
                # The chatbot must be able to get a list of other participants in the channel.
                "users": self.handle_users,
                "forget": self.handle_forget,
                # As a bare minimum conversation starter, the chatbot must respond to a “hello” utterance directed to it, with another hello to the same source that greeted it first. 
                # If the chatbot itself had initiated the greeting, it must not respond to the response.
                "hello": self.handle_hello,
            }

            if command_name in BASE_COMMANDS:
                BASE_COMMANDS[command_name](conn, self.channel, author)
                return
            else:
                self.on_pubmsg_personalities(command_text, author)
            
    
    def on_pubmsg_personalities(self, command, author=None):
        if self.current_personality:
            # Sheldon Commands
            if self.current_personality.get_name() == 'sheldon':
                print("Routing message to Sheldon personality")
                self.current_personality.generate_wiki_response(command, author=author)



if __name__ == "__main__":
    # The chatbot’s name must end with the string “-bot”.
    bot_nickname = "Sheldon-AS-bot"
    PersonalityBot(CHANNEL, bot_nickname).start()
