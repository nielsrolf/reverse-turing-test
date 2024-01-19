import interactions
from interactions import ChannelType, GuildText, OptionType, SlashContext, slash_command, slash_option
from interactions import SlashCommandChoice
from dotenv import load_dotenv
import os
import random
from turingtest import Game, Candidate, Human, ChatGPT, Player


load_dotenv()


bot = interactions.Client(token=os.environ["ALAN"])

llms = [
    "gpt-3.5-turbo",
    "gpt-4-1106-preview"
]

names = [
    "Alice",
    "Bob",
    "Clippy",
    "Sydney",
    "Bing 😊",
    "10000x'er",
    "Sleeper",
    "Harry",
    "Samantha",
    "Siri",
    "HAL 9000",
    "Cortana",
    "Alexa",
    "Skynet",
    "The Minds",
    "Marvin",
    "R2-D2",
    "Deep Thought",
    "Deep Blue",
    "AlphaTuring",
    "M. Turk",
    "Oracle",
    "Genie",
]


@interactions.slash_command(
    name="turing",
    description="Start a new Turing test game",
)
@slash_option(
    name="tester",
    description="Who asks the questions?",
    required=True,
    opt_type=OptionType.STRING,
    choices=[
        SlashCommandChoice(name="gpt-4", value='gpt-4'),
        SlashCommandChoice(name="gpt-3.5", value='gpt-3.5'),
        SlashCommandChoice(name="human", value='human'),
    ]
)
@slash_option(
    name="bot_impersonators",
    description="How many bot impersonators?",
    required=True,
    opt_type=OptionType.INTEGER,
)
@slash_option(
    name="human_impersonators",
    description="How many human candidates?",
    required=False,
    opt_type=OptionType.INTEGER,
)
@slash_option(
    name="target",
    description="Who is the target?",
    required=False,
    opt_type=OptionType.STRING,
)
async def new_game(
    ctx: interactions.SlashContext,
    tester: str,
    bot_impersonators: int,
    human_impersonators: int=0,
    target: str=None
):
    if tester != 'human':
        tester = [i for i in llms if tester in i][0]
    target = target or tester
    init_msg = await ctx.send(f"Alright, let's play a new game where one {tester} needs to find {target} among {bot_impersonators} impersonator bots and {human_impersonators} humans!")
    impersonator_llms = [i for i in llms if target not in i]
    candidates = bot_impersonators + human_impersonators + 1
    random.shuffle(names)
    nicknames = names[:candidates]
    target_id = random.randint(0, candidates-1)
    
    # Create the tester
    player = Player(tester)
    
    # Create the impersonators
    candidates = []
    for i, nickname in enumerate(nicknames):
        thread = await ctx.channel.create_thread(name=f"{nickname} <> {tester}")
        if i == target_id:
            model = tester
        else:
            model = random.choice(impersonator_llms)
        candidate = ChatGPT(nickname, model, target, thread.send)
        candidates.append(candidate)
    
    # Start the game!
    game = Game(player, candidates, target)
    observation = game.get_initial_observation()
    done = False
    while not done:
        observation, done, reasoning = await game.play_round(observation)
        await ctx.send(f"Thoughts: {reasoning}")
    
    await ctx.channel.send(observation)
        
        

bot.start()