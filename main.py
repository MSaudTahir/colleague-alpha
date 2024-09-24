import discord
import os # default module
from dotenv import load_dotenv

from llm.google_handler import Google_LLM
from llm.openai_handler import OpenAI_LLM

load_dotenv() # load all the variables from the env file
bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name="hello", description="Say hello to the bot")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")

@bot.command(description="Sends the bot's latency.") # this decorator makes a slash command
async def ping(ctx: discord.ApplicationContext): # a slash command will be created with the name "ping"
    await ctx.respond(f"Pong! Latency is {bot.latency}")

ai = discord.SlashCommandGroup("ai", "AI commands")

@ai.command()
async def query_google(interaction: discord.Interaction, model: str, prompt: str):
    google_llm = Google_LLM(model)
    await interaction.response.defer()
    await interaction.followup.send(google_llm.get_response(prompt=prompt))

@ai.command()
async def query_openai(interaction: discord.Interaction, model: str, prompt: str):
    openai_llm = OpenAI_LLM(model)
    await interaction.response.defer()
    await interaction.followup.send(openai_llm.get_response(prompt=prompt))

bot.add_application_command(ai)

bot.run(os.getenv('DISCORD_BOT_TOKEN')) # run the bot with the token