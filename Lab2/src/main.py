from irc.bot import SingleServerIRCBot
import sys
import spacy

# CHANNEL = "#CSC582"
CHANNEL = "#CSC582Testing"

class PersonalityBot(SingleServerIRCBot):
    SERVER = 'irc.libera.chat'
    PORT = 6667

    def __init__(self, channel, nickname):
        super().__init__([(self.SERVER, self.PORT)], nickname, nickname)
        self.channel = channel
        self.nickname = nickname
        self.knowledge = {}
    
    def on_welcome(self, conn, event):
        self.conn = conn
        conn.join(self.channel)
    
    def on_join(self, conn, event):
        if event.source.nick == conn.get_nickname():
            print("hello")
            conn.privmsg(self.channel, f"I have joined the channel!")
    
    # That chatbot is expected to respond to any utterance in the channel that begins with its name followed by an immediate colon (:) symbol.
    def parse_privmsg(self, text, botnick, channel):
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
        conn.privmsg(
            channel,
         (f"{author} I am a chatbot with multiple different personalities!")
        )
    
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

    
    def on_pubmsg(self, conn, event):
        if not event.arguments:
            return
        
        text = event.arguments[0]
        author = event.source.nick
        
        if author == conn.get_nickname():
            return
        
        # Commands take priority - check if this is a command first
        command_text = self.parse_privmsg(text, self.nickname, self.channel)
        if command_text:
            command_lower = command_text.lower().strip()
            
            # Get first word for single-word commands
            parts = command_text.split(None, 1)
            command_name = parts[0].lower()
            command_query = parts[1] if len(parts) > 1 else ""
            
            COMMANDS = {
                "die": self.handle_die,
                "users": self.handle_users,
                "forget": self.handle_forget,
                "who are you?": self.handle_usage,
                "usage": self.handle_usage,
                "hello": self.handle_hello,
            }

            print(command_name)
            
            # The chatbot must kill itself when command “die” is given to it (preceded by its name followed by colon).
            if command_name == "die":
                self.handle_die(conn, self.channel, author)
                return
            # The chatbot must be able to get a list of other participants in the channel.
            elif command_name == "users":
                self.handle_users(conn, self.channel, author)
                return
            elif command_name == "who":
                self.handle_who_are_you(conn, self.channel, author)
                return      
            # As a bare minimum conversation starter, the chatbot must respond to a “hello” utterance directed to it, with another hello to the same source that greeted it first. 
            # If the chatbot itself had initiated the greeting, it must not respond to the response.
            elif command_name == "hello":
                self.handle_hello(conn, self.channel, author)
                return          



if __name__ == "__main__":
    # The chatbot’s name must end with the string “-bot”.
    bot_nickname = "ASP-bot"
    PersonalityBot(CHANNEL, bot_nickname).start()

