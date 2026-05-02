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
import requests
import spacy
nlp = spacy.load("en_core_web_sm")

WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "light showers", 81: "showers", 82: "heavy showers",
    95: "thunderstorm",
}

CITIES = [
    "San Luis Obispo", "Los Angeles", "San Francisco", "San Diego", "Sacramento",
    "San Jose", "Oakland", "Fresno", "Long Beach", "Bakersfield",
    "Anaheim", "Santa Ana", "Riverside", "Stockton", "Irvine",
    "Chula Vista", "Fremont", "Santa Barbara", "Modesto", "Oxnard",
    "Fontana", "Moreno Valley", "Glendale", "Santa Clarita", "Huntington Beach",
    "Berkeley", "Pasadena", "Pomona", "Escondido", "Torrance",
    "New York", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "Dallas", "Austin", "Seattle", "Denver",
    "Boston", "Atlanta", "Miami", "Las Vegas", "Portland",
    "Nashville", "Detroit", "Memphis", "Baltimore", "Milwaukee",
    "London", "Paris", "Tokyo", "Beijing", "Shanghai",
    "Mumbai", "Delhi", "Bangalore", "Dubai", "Sydney",
    "Melbourne", "Toronto", "Vancouver", "Mexico City", "São Paulo",
    "Buenos Aires", "Rio de Janeiro", "Cairo", "Lagos", "Nairobi",
    "Istanbul", "Moscow", "Berlin", "Madrid", "Rome",
    "Amsterdam", "Brussels", "Vienna", "Stockholm", "Oslo",
    "Copenhagen", "Zurich", "Singapore", "Bangkok", "Seoul",
    "Hong Kong", "Taipei", "Jakarta", "Kuala Lumpur", "Manila",
    "Karachi", "Lahore", "Dhaka", "Colombo", "Kathmandu",
]

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
        self.knowledge[nick] = f"joined at {join_time}"
        self.say(f"Oh... Did {nick} just join this channel? They did right? Someone please tell me they did.")

    def on_user_left(self, nick):
        left_time = datetime.now().strftime("%H:%M:%S")
        self.knowledge[nick] = f"left at {left_time}"
        self.say(f"W-wait... did {nick} just leave? They're gone aren't they... aren't they?")

    def get_who_left(self):
        left = {nick: status for nick, status in self.knowledge.items() if status.startswith("left")}
        if left:
            summary = ", ".join(f"{nick} ({status})" for nick, status in left.items())
            self.say(f"S-so... the people who left are... {summary}... that's right isn't it?")
        else:
            self.say("I-I don't think anyone has left yet... or did they? Did someone leave and I missed it?!")
    
    def current_day_and_time(self):
        now = datetime.now()
        formatted_now = now.strftime("%A, %B %d, %Y %H:%M:%S")
        self.say(f"T-the current d-day and time is... Oh I think it's this: {formatted_now}... I hope i'm not wrong ")
    
    def parse_city(self, command):
        doc = nlp(command)
        for ent in doc.ents:
            if ent.label_ == "GPE":
                return ent.text
        return None

    def get_weather(self, command):
        city = self.parse_city(command)
        print(city)
        if city:
            geo = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1}
            ).json()

            loc = geo['results'][0]
            lat, lon = loc['latitude'], loc['longitude']

            weather = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    "current_weather": True,
                    "temperature_unit": "fahrenheit"
                }
            ).json()

            cw = weather['current_weather']
            condition = WEATHER_CODES.get(cw["weathercode"], "unknown conditions")
            self.say(f"Oh the cc-current weather in {city}, i-is is.... Temp: {cw['temperature']}°F, and {condition}. But I can't go o-outside and check so i'm not 100% s-s-sure :(")
        else:
            self.say("W-wait... where exactly? I couldn't figure out the city... did you mention one? You did right?")


    def personality_tick(self):
        options = [
            self.current_day_and_time,
            lambda: self.get_weather(random.choice(CITIES))
        ]

        random.choice(options)()