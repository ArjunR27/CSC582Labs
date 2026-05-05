from irc.bot import SingleServerIRCBot
import sys
from archetypes import sheldon
import random
from collections import deque, defaultdict
import time
import re


# CHANNEL = "#CSC582"
CHANNEL = "#CSC582TestTest"
BOT_DESCRIPTION = (
    "I am Sheldon-ASP-bot. I answer science and tech questions with Wikipedia-grounded facts."
)

class PersonalityBot(SingleServerIRCBot):
    SERVER = 'irc.libera.chat'
    PORT = 6667
    ALLOWED_PERSONALITIES = set(['sheldon', 'normal'])
    BOT_INTERJECTION_COOLDOWN_SEC = 45

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
        self.last_bot_interjection_ts = 0.0
    
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
        self.check_and_interject_bot_to_bot()
        if self.current_personality:
            self.current_personality.personality_tick()
        self._schedule_tick()

    def check_and_interject_bot_to_bot(self):
        now = time.time()
        if now - self.last_bot_interjection_ts < self.BOT_INTERJECTION_COOLDOWN_SEC:
            return

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
            return

        chosen_pair = random.choice(active_pairs)
        target = random.choice(list(chosen_pair))
        self.conn.privmsg(self.channel, f"{target}: {self.random_fact()}")
        self.last_bot_interjection_ts = now
    
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
        conn.privmsg(channel, f"Hi {author}!")

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
            command_lower = command_text.lower().strip()
            normalized_command = re.sub(r"[^a-z0-9\s]", "", command_lower)

            if normalized_command in {"what do you do", "so what do you do"}:
                self.handle_bot_description(conn, self.channel, author)
                return
            
            # Get first word for single-word commands
            parts = command_text.split(None, 1)
            command_name = parts[0].lower()
            command_query = parts[1] if len(parts) > 1 else ""

            print(command_name, command_query)
            
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
                self.on_pubmsg_personalities(command_text)
            
    
    def on_pubmsg_personalities(self, command):
        if self.current_personality:
            # Sheldon Commands
            if self.current_personality.get_name() == 'sheldon':
                print("here")
                self.current_personality.generate_wiki_response(command)



if __name__ == "__main__":
    # The chatbot’s name must end with the string “-bot”.
    bot_nickname = "Sheldon-AS-bot"
    PersonalityBot(CHANNEL, bot_nickname).start()
