"""
music / poetry obsessed
- shy
- frequently sings, volunteers lyrics (lyrics lines delimited by “/” )
- responds by singing: to people’s random conversation if they match some lyrics the bot knows.
- poses rhetorical questions about the band information that it knows about. 
- can also answer questions about the lyrics (who wrote it? what band? what year?)

Data Source:
- Lyrics database
- Band / song database
"""
class Tweety():
    def __init__(self, conn, channel, bot):
        self.name = 'tweety'
        self.conn = conn
        self.channel = channel
        self.bot = bot
    
    def get_name(self):
        return self.name
    
    def say(self, msg):
        self.conn.privmsg(self.channel, msg)
    
    def personality_tick(self):
        return