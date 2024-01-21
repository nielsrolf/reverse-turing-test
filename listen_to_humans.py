import discord
from dotenv import load_dotenv
import os
import asyncio
import threading


load_dotenv()

intents = discord.Intents.default()
intents.messages = True  # Enable messages intents
intents.message_content = True
intents.guild_messages = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

    
thread_listeners = {}
class ThreadListener:
    def __init__(self, channel):
        self.channel = channel
        self.history = []
        thread_listeners[channel.id] = self
    
    async def wait_for(self, check=lambda message: True):
        ignore = len(self.history)
        while len(messages := [message for message in self.history[ignore:] if check(message)]) == 0:
            print(self.history)
            await asyncio.sleep(1)
        return messages[0]


@client.event
async def on_message(message):
    if message.channel.id in thread_listeners:
        print("Got message from", message.author)
        thread_listeners[message.channel.id].history.append(message)
        

def start_bot():
    client.run(os.getenv('TURING'))

bot_thread = threading.Thread(target=start_bot)
bot_thread.start()
