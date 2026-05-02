"""
Personality
insecure, repetitive, needs constant reassurance
- very talkative
- volunteers information unsolicited
- retelling of trivial facts
- asks very obvious questions
- restates obvious facts and asks for confirmation

Data Source
- chatroom events (who has entered, what time, who has left, user names)
- current time, day, date, weather
"""

from datetime import datetime
import random

class Angel():
    def __init__(self, conn, channel, bot):
        self.name = 'angel'
        self.conn = conn
        self.channel = channel
        # References the IRC PersonalityBot class that we created
        self.bot = bot
        self.knowledge = {}
    
    def get_name(self):
        return self.name
    
    def say(self, msg):
        self.conn.privmsg(self.channel, msg)

    def on_user_joined(self, nick):
        join_time = datetime.now().strftime("%H:%M:%S")
        self.knowledge[nick] = join_time
        self.say(f"Oh... Did {nick} just join this channel? They did right? Someone please tell me they did.")
    
    def current_day_and_time(self):
        now = datetime.now()
        formatted_now = now.strftime("%A, %B %d, %Y %H:%M:%S")
        self.say(f"T-the current d-day and time is... Oh I think it's this: {formatted_now}... I hope i'm not wrong ")

    
    def personality_tick(self):
        options = [
            self.current_day_and_time
        ]

        random.choice(options)()