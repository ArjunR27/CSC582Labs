"""
politician, interested in just repeating his talking points
- well defined platform made up of paragraph length positions on 20-30 topics 
- always steers conversation to something on his platform for which he has prepared remarks
- frequently attacks his unnamed opponent, blaming everything on them
- volunteers leading, often nonsensical, yes/no questions just to prompt his responses
- fake expressions of sympathy and gestures

Data Source
- corpus of political positions (may use real examples or combination of multiple platforms)
"""

class Quimby():
    def __init__(self, conn, channel, bot):
        self.name = 'quimby'
        self.conn = conn
        self.channel = channel
        self.bot = bot
    
    def get_name(self):
        return self.name
    
    def say(self, msg):
        self.conn.privmsg(self.channel, msg)
    
    def personality_tick(self):
        return