from irc.bot import SingleServerIRCBot
import sys
from archetypes import angel, guss, abraham, quimby, sheldon, tweety
import random


# CHANNEL = "#CSC582"
CHANNEL = "#CSC582Test"

class PersonalityBot(SingleServerIRCBot):
    SERVER = 'irc.libera.chat'
    PORT = 6667
    ALLOWED_PERSONALITIES = set(['angel', 'guss', 'tweety', 'sheldon', 'abraham', 'quimby', 'normal'])

    def __init__(self, channel, nickname):
        super().__init__([(self.SERVER, self.PORT)], nickname, nickname)
        self.channel = channel
        self.nickname = nickname
        self.knowledge = {}
        self.current_personality = None
        self.PERSONALITIES = {}
    
    def on_welcome(self, conn, event):
        self.conn = conn
        conn.join(self.channel)
        self._schedule_tick()
    
    # Schedules the personalities own 'tick' every 3-5 seconds
    # will change depending on the current personality
    def _schedule_tick(self):
        # can change the delay
        delay = random.uniform(25, 30)
        self.reactor.scheduler.execute_after(delay, self._personality_tick)

    def _personality_tick(self):
        if self.current_personality:
            self.current_personality.personality_tick()
        self._schedule_tick()
    
    def on_join(self, conn, event):
        if event.source.nick == conn.get_nickname():
            print("hello")
            conn.privmsg(self.channel, f"I have joined the channel!")
            return

        if self.current_personality:
            match self.current_personality.get_name():
                case 'angel':
                    self.current_personality.on_user_joined(event.source.nick)

    def on_part(self, conn, event):
        nick = event.source.nick
        if nick == conn.get_nickname():
            return
        if self.current_personality:
            match self.current_personality.get_name():
                case 'angel':
                    self.current_personality.on_user_left(nick)

    def on_quit(self, conn, event):
        nick = event.source.nick
        if nick == conn.get_nickname():
            return
        if self.current_personality:
            match self.current_personality.get_name():
                case 'angel':
                    self.current_personality.on_user_left(nick)
    
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
    
    def handle_usage(self, conn, channel, author):
        self.handle_who_are_you(conn, channel, author)
    
    def handle_who_are_you(self, conn, channel, author):
        conn.privmsg(channel, f"{author}: My name is {self.nickname}. I was created by Sid and Arjun in CSC-582")
        conn.privmsg(channel, "I am a chatbot with multiple different personalities!")
        conn.privmsg(channel, "Personalities: Angel, Guss, Tweety, Sheldon, Abraham, Quimby")
        conn.privmsg(channel, "To switch, type: ASP-bot: switch [name]")
        
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
    
    def handle_switch(self, conn, channel, personality):
        if personality.lower() in self.PERSONALITIES:
            self.current_personality = self.PERSONALITIES[personality]
        else:
            if personality.lower() in self.ALLOWED_PERSONALITIES:
                if personality.lower() == 'angel':
                    conn.privmsg(channel, "Requested a change to Angel!")
                    self.current_personality = angel.Angel(conn, channel, self)
                    self.PERSONALITIES[personality] = self.current_personality
                elif personality.lower() == 'normal':
                    conn.privmsg(channel, 'Requested a change to base personality!')
                    self.current_personality = None
                elif personality.lower() == 'guss':
                    conn.privmsg(channel, "Requested a change to Guss!")
                    self.current_personality = guss.Guss(conn, channel, self)
                    self.PERSONALITIES[personality] = self.current_personality
                elif personality.lower() == 'abraham':
                    conn.privmsg(channel, "Requested a change to Abraham!")
                    self.current_personality = abraham.Abraham(conn, channel, self)
                    self.PERSONALITIES[personality] = self.current_personality
                elif personality.lower() == 'quimby':
                    conn.privmsg(channel, "Requested a change to Quimby!")
                    self.current_personality = quimby.Quimby(conn, channel, self)
                    self.PERSONALITIES[personality] = self.current_personality
                elif personality.lower() == 'sheldon':
                    conn.privmsg(channel, "Requested a change to Sheldon!")
                    self.current_personality = sheldon.Sheldon(conn, channel, self)
                    self.PERSONALITIES[personality] = self.current_personality
                elif personality.lower() == 'tweety':
                    conn.privmsg(channel, "Requested a change to Tweety!")
                    self.current_personality = tweety.Tweety(conn, channel, self)
                    self.PERSONALITIES[personality] = self.current_personality


    def on_pubmsg(self, conn, event):
        if not event.arguments:
            return
        
        text = event.arguments[0]
        author = event.source.nick
        
        if author == conn.get_nickname():
            return
        
        # Commands take priority - check if this is a command first
        command_text = self.parse_privmsg(conn, text, self.nickname, self.channel)
        if command_text:
            command_lower = command_text.lower().strip()
            
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
                "usage": self.handle_usage,
                # As a bare minimum conversation starter, the chatbot must respond to a “hello” utterance directed to it, with another hello to the same source that greeted it first. 
                # If the chatbot itself had initiated the greeting, it must not respond to the response.
                "hello": self.handle_hello,
            }

            if command_name in BASE_COMMANDS:
                BASE_COMMANDS[command_name](conn, self.channel, author)
                return
            elif command_name == "switch":
                self.handle_switch(conn, self.channel, command_query)
            else:
                self.on_pubmsg_personalities(command_text)
            
    
    def on_pubmsg_personalities(self, command):
        if self.current_personality:
            
            # Angel Commands
            if self.current_personality.get_name() == 'angel':
                if 'weather' in command:
                    self.current_personality.get_weather(command)
                elif 'left' or 'leave' in command.lower():
                    self.current_personality.get_who_left()
            
            # Guss Commands

            # Tweety Commands

            # Sheldon Commands

            # Abraham Commands

            # Quimby Commands





if __name__ == "__main__":
    # The chatbot’s name must end with the string “-bot”.
    bot_nickname = "ASP-bot"
    PersonalityBot(CHANNEL, bot_nickname).start()

