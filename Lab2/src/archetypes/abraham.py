"""
bitter old man (American background)
- volunteers reminiscences about old days 
- will often “get reminded” of “a little story” and start telling it in multiple lines. But sometimes will forget and won’t finish story
- greatest time was the 1950’s and early 60’s until “Rock and Roll ruined everything”

Data Source:
- short corpus of stories and memories
- actual info from the Simpsons character is fine. 
"""
class Abraham():
    def __init__(self, conn, channel, bot):
        self.name = 'abraham'
        self.conn = conn
        self.channel = channel
        self.bot = bot
    
    def get_name(self):
        return self.name
    
    def say(self, msg):
        self.conn.privmsg(self.channel, msg)
    
    def personality_tick(self):
        return