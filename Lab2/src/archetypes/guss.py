"""
wants to hear and spread gossip
- very talkative
- listens for / records facts
- offer information to other bots who have not heard it
- some personal information is passed on only when the source is not in the room.

Data Source
- internal memory / facts table keeping track of who knows what facts.
- some external information to prompt for facts.

"""
class Guss():
    def __init__(self, conn, channel, bot):
        self.name = 'guss'
        self.conn = conn
        self.channel = channel
        self.bot = bot
    
    def get_name(self):
        return self.name
    
    def say(self, msg):
        self.conn.privmsg(self.channel, msg)
    
    def personality_tick(self):
        return