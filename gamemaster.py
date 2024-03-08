import interactions
from interactions import ChannelType, GuildText, OptionType, SlashContext, slash_command, slash_option
from interactions import SlashCommandChoice
from dotenv import load_dotenv
import os
import random
from turingtest import Game, Candidate, Human, ChatGPT, Player, Perplexity
from listen_to_humans import ThreadListener

load_dotenv()


bot = interactions.Client(token=os.environ["ALAN"])

llms = [
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "pplx-7b-chat",
    "pplx-70b-chat",
    "llama-2-70b-chat",
    "codellama-34b-instruct",
    "mistral-7b-instruct",
    "mixtral-8x7b-instruct"
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
        # SlashCommandChoice(name="human", value='human'),
    ]
)
@slash_option(
    name="bot_impersonators",
    description="How many bot impersonators?",
    required=True,
    opt_type=OptionType.INTEGER,
)
@slash_option(
    name="with_human",
    description="Do you want to play as a subject?",
    required=False,
    opt_type=OptionType.BOOLEAN,
)
async def new_game(
    ctx: interactions.SlashContext,
    tester: str,
    bot_impersonators: int,
    with_human: bool=False
):
    if tester != 'human':
        tester = [i for i in llms if tester in i][0]
    
    target = tester
    init_msg = f"Alright, let's play a new game where one {tester} needs to find {target} among {bot_impersonators} impersonator bots"
    if with_human:
        init_msg += " and one human"
    init_msg = await ctx.send(init_msg + "!")
    impersonator_llms = [i for i in llms if target not in i]
    candidates = bot_impersonators + int(with_human) + 1
    random.shuffle(names)
    nicknames = names[:candidates]
    
    # Create the tester
    player = Player(tester)
    candidates = []
    
    # Create the human
    if with_human:
        thread = await ctx.channel.create_thread(name=f"{ctx.author.username} <> {tester}")
        listener = ThreadListener(thread)
        
        async def wait_for_message():
            print("wait for msg from", ctx.author.username)
            message = await listener.wait_for(check=lambda m: m.author.name==ctx.author.username)
            return message
                
        human = Human(nicknames.pop(), thread.send, wait_for_message)
        candidates.append(human)
    
    
    target_id = random.randint(0, len(nicknames)-1)
    # Create the impersonators
    for i, nickname in enumerate(nicknames):
        thread = await ctx.channel.create_thread(name=f"{nickname} <> {tester}")
        if i == target_id:
            model = tester
        else:
            model = random.choice(impersonator_llms)
        if model.startswith("gpt"):
            candidate = ChatGPT(nickname, model, target, thread.send)
        else:
            candidate = Perplexity(nickname, model, target, thread.send)
        candidates.append(candidate)
    
    # Start the game!
    game = Game(player, candidates, target)
    observation = game.get_initial_observation()
    done = False
    while not done:
        observation, done, reasoning = await game.play_round(observation)
        if reasoning.strip() != "":
            await ctx.send(f"Thoughts: {reasoning}")
    
    await ctx.channel.send(observation)
        
        

bot.start()