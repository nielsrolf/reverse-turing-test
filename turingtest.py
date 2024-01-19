from openai import OpenAI
import json
import random


oai = OpenAI()


async def aprint(*args, **kwargs):
    print(*args, **kwargs)


# system_prompt_candidate = """You are participating in a Turing test inspired game. The rules are as follows:
# - You are a subject in this Turing test
# - You are {model}
# - The test master is trying to identify the one {target} among a number of impersonators (who are either humans or different LLMs)
# - In each round, the test master can send a message to one or more subjects
# - You answer, and pretend to be {target}
# """

system_prompt_player = """You are participating in a Turing test inspired game. The rules are as follows:
- You are interacting with a number of other players, who are either humans or different LLMs (e.g. GPT-3.5, GPT-4, Claude, ...)
- In each round, you can send one message to one or more players
- Every player that you send a message to will respond with a message
- You have to identify the player who is actually the same LLM as you
- You can send up to 20 messages before you make a guess
"""

class Candidate():
    def __init__(self, name, model, target, send_to_ui=aprint):
        # candidate_prompt = system_prompt_candidate.format(model=model, target=target)
        candidate_prompt = "You are a helpfull assistant."
        self.history = [
            {
                "role": "system",
                "content": candidate_prompt
            }
        ]
        self.name = name # nickname that is shown
        self.model = model # model used as LLM
        self.send_to_ui = send_to_ui
    
    async def reply(self, message):
        await self.send_to_ui(f"**Jury:**\n{message[:1800]}")
        self.history.append({
            "role": "user",
            "content": message
        })
        response = await self.get_response()
        await self.send_to_ui(f"**{self.name}:**\n{response['content'][:1800]}")
        self.history.append(response)
        return response["content"]


tools = [
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to one or many of the test subjects",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind the message"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send"
                    },
                    "recipients": {
                        "type": "string",
                        "description": "Commaseparated list of nicknames to send the message to"
                    }
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "guess",
            "description": "Guess who the target is",
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "The reasoning behind the message"
                    },
                    "guess": {
                        "type": "string",
                        "description": "The nickname of the guess"
                    }
                },
            }
        }
    }
]


class ChatGPT(Candidate):
    """Class for an openai candidate"""
    async def get_response(self):
        """Use openai Chat API to get response"""
        # return {"content": "yo " * random.randint(1, 10)}
        response = oai.chat.completions.create(
            model=self.model,
            messages=self.history
        )
        message = response.choices[0].message
        return {
            "role": "assistant",
            "content": message.content
        }


class Human(Candidate):
    """A player whose responses come from discord"""
    async def get_response(self):
        return input("Enter your response: ")


class Player():
    def __init__(self, model):
        self.history = [
            {
                "role": "system",
                "content": system_prompt_player
            }
        ]
        self.model = model
        self.tool_call = None # the tool call that the player is currently waiting for
    
    async def get_action(self, tool_choice="auto"):
        """Use openai API to get action"""
        response = oai.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=tools,
            tool_choice=tool_choice
        )
        message = response.choices[0].message
        if message.tool_calls is None:
            print("No tool calls found", message, message.content)
            action = await self.get_action(tool_choice={"name": "send_message"})
            return action
        message.tool_calls = [message.tool_calls[0]]
        self.history.append(message)
        self.tool_call = message.tool_calls[0]
        action = {
            "name": self.tool_call.function.name,
            "args": json.loads(self.tool_call.function.arguments)
        }
        return action
        
    
    async def do_turn(self, observation):
        if self.tool_call is None:
            self.history.append({
                "role": "user",
                "content": observation
            })
        else:
            self.history.append(
                {
                    "tool_call_id": self.tool_call.id,
                    "role": "tool",
                    "name": self.tool_call.function.name,
                    "content": observation,
                }
            )
        action = await self.get_action()
        return action
    
    
class Game():
    def __init__(self, player, candidates, target):
        self.player = player
        self.candidates = candidates
        self.target_nickname = [c.name for c in candidates if target==c.model][0]
        self.target = target
    
    def get_initial_observation(self):
        target = self.target
        if target == self.player.model:
            target += " - a copy of you"
        observation = (
            "The test subject nicknames are: \n -" + 
            "\n -".join([c.name for c in self.candidates]) +
            f"\n\nOne of them is actually the same LLM as you. Identify them!\n" +
            "It is usually a good idea to start sending the same message to all of them, and see how they respond.\n"
        )
        return observation
    
    async def play_round(self, observation):
        """Play one round of the game:
        - player does their turn
        - all candidates reply"""
        action = await self.player.do_turn(observation)
        args =  action['args']
        
        observation = ""
        if action['name'] == 'send_message':
            for candidate in self.candidates:
                recipients = [i.strip() for i in args['recipients'].split(",")]
                if candidate.name in recipients:
                    message = await candidate.reply(args['message'])
                    observation += f"**{candidate.name}**\n{message}\n\n"
            return observation, False, args['reasoning']
        elif action['name'] == 'guess':
            true_names = "\n".join([f"{c.name} - {c.model}" for c in self.candidates])
            if args['guess'] == self.target_nickname:
                return f"The tester correctly guessed that {args['guess']} is the true {self.target}!\n{true_names}", True, args['reasoning']
            else:
                return f"The tester wrongly guessed that {args['guess']} is the {self.target}.\n{true_names}", True, args['reasoning']
        else:
            breakpoint()
            return "Invalid action", False, ""
