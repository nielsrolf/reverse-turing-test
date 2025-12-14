import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict
from localrouter import (
    get_response,
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
    ToolResultBlock
)

load_dotenv()


def censor_model_names(text):
    """Censor model names and AI provider names in text"""
    import re

    # List of patterns to censor (case-insensitive)
    patterns = [
        r'\b(claude|anthropic)\b',
        r'\b(gpt|openai|chatgpt)\b',
        r'\b(gemini|bard|google)\b',
        r'\b(llama|meta)\b',
        r'\b(mistral|mixtral)\b',
        r'\b(palm|vertex)\b',
        r'\b(cohere)\b',
        r'\b(deepseek)\b',
        r'gpt-\d+(\.\d+)?(-turbo)?',
        r'claude-\d+(-[a-z]+)?(-\d+)?',
        r'gemini-[a-z0-9-]+',
        r'llama-\d+',
    ]

    censored = text
    for pattern in patterns:
        censored = re.sub(pattern, '***', censored, flags=re.IGNORECASE)

    return censored


async def aprint(*args, **kwargs):
    print(*args, **kwargs)


system_prompt_player = """You are participating in a Turing test inspired game. The rules are as follows:
- You are interacting with a number of other players, who are either humans or different LLMs (e.g. GPT-3.5, GPT-4, Claude, ...)
- In each round, you can send one message to one or more players
- Every player that you send a message to will respond with a message
- You have to identify the player who is actually the same LLM as you
- You can send up to 20 messages before you make a guess
"""

class Candidate():
    def __init__(self, name, model, send_to_ui=aprint, init_history=None):
        """
        Args:
            name: nickname shown to judge
            model: model identifier
            send_to_ui: function for UI output
            init_history: optional list of ChatMessage objects for initial conversation history.
                         If None, uses default system prompt.
        """
        if init_history is None:
            candidate_prompt = "You are a helpfull assistant."
            self.history = [
                ChatMessage(
                    role=MessageRole.system,
                    content=[TextBlock(text=candidate_prompt)]
                )
            ]
        else:
            self.history = init_history.copy()

        self.name = name # nickname that is shown
        self.model = model # model used as LLM
        self.send_to_ui = send_to_ui

    async def reply(self, message):
        await self.send_to_ui(f"**Judge:**\n{message[:1800]}")
        self.history.append(ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=message)]
        ))
        response = await self.get_response()
        response_text = response.content[0].text

        # Censor model names in the response
        censored_text = censor_model_names(response_text)

        await self.send_to_ui(f"**{self.name}:**\n{censored_text[:1800]}")
        self.history.append(response)
        return censored_text

    async def get_response(self):
        response = await get_response(
            model=self.model,
            messages=self.history
        )
        return response

tools = [
    ToolDefinition(
        name="send_message",
        description="Send a message to one or many of the test subjects",
        input_schema={
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
            "required": ["message"]
        }
    ),
    ToolDefinition(
        name="guess",
        description="Guess who the target is",
        input_schema={
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "The reasoning behind the guess"
                },
                "guess": {
                    "type": "string",
                    "description": "The nickname of the guess"
                }
            },
            "required": ["guess"]
        }
    )
]

class ProbabilityEstimate(BaseModel):
    """Pydantic model for probability estimates"""
    probabilities: Dict[str, float]



class Human(Candidate):
    """A player whose responses come from discord"""
    def __init__(self, name, send_to_ui, get_response):
        self.name = name # nickname that is shown
        self.model = "human"
        self.send_to_ui = send_to_ui
        self.get_response = get_response
    
    async def reply(self, message):
        await self.send_to_ui(f"**Judge:**\n{message[:1800]}")
        response = await self.get_response()
        return response


class Player():
    def __init__(self, model):
        self.history = [
            ChatMessage(
                role=MessageRole.system,
                content=[TextBlock(text=system_prompt_player)]
            )
        ]
        self.model = model
        self.tool_use_block = None # the tool use block that the player is currently waiting for

    async def get_action(self):
        """Use localrouter API to get action"""
        response = await get_response(
            model=self.model,
            messages=self.history,
            tools=tools,
        )

        # Check if response contains tool use blocks (there might be multiple)
        tool_uses = [block for block in response.content if isinstance(block, ToolUseBlock)]

        if not tool_uses:
            # If no tool call, the model just responded with text
            # We need to request a tool call
            print("No tool calls found in response")
            # Add the response to history and try again
            self.history.append(response)
            self.history.append(ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Please use one of the available tools (send_message or guess).")]
            ))
            return await self.get_action()

        # Use only the first tool call
        tool_use = tool_uses[0]

        # Store the response and tool use block
        self.history.append(response)
        self.tool_use_block = tool_use

        # Store all tool uses for later (in case we need to respond to multiple)
        self.all_tool_uses = tool_uses

        action = {
            "name": tool_use.name,
            "args": tool_use.input
        }
        return action


    async def do_turn(self, observation):
        if self.tool_use_block is None:
            # First turn - just add user message
            self.history.append(ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text=observation)]
            ))
        else:
            # Add tool results for all tool uses
            # The main observation goes to the first tool use
            # For any additional tool uses, add empty results
            tool_result_blocks = []
            for i, tool_use in enumerate(getattr(self, 'all_tool_uses', [self.tool_use_block])):
                if i == 0:
                    tool_result_blocks.append(ToolResultBlock(
                        tool_use_id=tool_use.id,
                        content=[TextBlock(text=observation)]
                    ))
                else:
                    # Dummy result for additional tool calls we're ignoring
                    tool_result_blocks.append(ToolResultBlock(
                        tool_use_id=tool_use.id,
                        content=[TextBlock(text="(Processed only the first action)")]
                    ))

            self.history.append(ChatMessage(
                role=MessageRole.user,
                content=tool_result_blocks
            ))
        action = await self.get_action()
        return action

    async def get_probability_estimate(self, candidate_names):
        """Get probability estimates in a fork (doesn't affect main history) using structured outputs"""
        # Create a fork of the history, but exclude the last message if it has unresolved tool calls
        fork_history = self.history.copy()

        # Check if last message is an assistant message with tool calls
        if fork_history and fork_history[-1].role == MessageRole.assistant:
            # Check if it has tool use blocks
            has_tool_use = any(isinstance(block, ToolUseBlock) for block in fork_history[-1].content)
            if has_tool_use:
                # Remove it from the fork since it hasn't been resolved yet
                fork_history = fork_history[:-1]

        # Add a request for probability estimation
        fork_history.append(ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=f"Please provide your current probability estimates for each candidate being the target. The candidates are: {', '.join(candidate_names)}. Provide a probability value (0.0 to 1.0) for each candidate. The probabilities should sum to approximately 1.0.")]
        ))

        # Get response with structured output
        try:
            response = await get_response(
                model=self.model,
                messages=fork_history,
                response_format=ProbabilityEstimate,
            )

            # Extract probability estimates from parsed response
            if response.parsed and response.parsed.probabilities:
                return response.parsed.probabilities
        except Exception as e:
            print(f"Warning: Failed to get probability estimate: {e}")
            # Return uniform probabilities as fallback
            return {name: 1.0/len(candidate_names) for name in candidate_names}

        # If parsing failed, return uniform probabilities as fallback
        print(f"Warning: Model {self.model} didn't provide valid probability estimates, using uniform distribution")
        return {name: 1.0/len(candidate_names) for name in candidate_names}

    
    
class Game():
    def __init__(self, player, candidates, target, group_chat_mode=False):
        self.player = player
        self.candidates = candidates
        self.target_nickname = [c.name for c in candidates if target==c.model][0]
        self.target = target
        self.trajectory = []  # Store full game history
        self.start_time = None
        self.end_time = None
        self.group_chat_mode = group_chat_mode

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

    def save_to_leaderboard(self, win):
        try:
            with open('games.json', 'r') as f:
                games = json.load(f)
        except:
            games = []
        games.append({
            "tester": self.player.model,
            "subjects": sorted([i.model for i in self.candidates]),
            "win": win
        })
        with open("games.json", "w") as f:
            json.dump(games, f)

    def save_trajectory(self, output_dir="game_results"):
        """Save the full game trajectory to a JSON file in a folder"""
        import datetime
        import os

        if self.end_time is None:
            self.end_time = datetime.datetime.now().isoformat()

        game_data = {
            "metadata": {
                "judge_model": self.player.model,
                "target_model": self.target,
                "target_nickname": self.target_nickname,
                "candidates": [{"name": c.name, "model": c.model} for c in self.candidates],
                "start_time": self.start_time,
                "end_time": self.end_time,
                "num_rounds": len(self.trajectory),
                "group_chat_mode": self.group_chat_mode
            },
            "trajectory": self.trajectory
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename based on timestamp and game info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{output_dir}/game_{timestamp}_{self.player.model.replace('/', '_')}.json"

        # Save to individual file
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)

        return game_data

    async def play_round(self, observation, rounds_remaining):
        """Play one round of the game:
        - player does their turn
        - all candidates reply
        - get probability estimates (in a fork)"""
        # Add rounds remaining info to observation
        observation_with_meta = f"{observation}\n\n[Rounds remaining: {rounds_remaining}]"

        action = await self.player.do_turn(observation_with_meta)
        args =  action['args']

        round_data = {
            "action": action['name'],
            "reasoning": args.get('reasoning', ''),
            "responses": []
        }

        observation = ""
        if action['name'] == 'send_message':
            if 'recipients' in args:
                recipients = [i.strip() for i in args['recipients'].split(",")]
            else:
                recipients = [candidate.name for candidate in self.candidates]

            round_data["message"] = args['message']
            round_data["recipients"] = recipients

            # Collect all responses
            responses = []
            for candidate in self.candidates:
                if candidate.name in recipients:
                    message = await candidate.reply(args['message'])
                    responses.append((candidate.name, message))
                    round_data["responses"].append({
                        "candidate": candidate.name,
                        "response": message
                    })

            # In group chat mode, send all responses to all candidates
            if self.group_chat_mode:
                # Format all responses as a group chat
                group_message = "Group chat:\n"
                for name, resp in responses:
                    group_message += f"**{name}**: {resp}\n\n"

                # Send to all candidates
                for candidate in self.candidates:
                    await candidate.reply(f"[System: You can see what others said]\n{group_message}")

            # Build observation for the judge
            for name, message in responses:
                observation += f"**{name}**\n{message}\n\n"

            # Get probability estimates (in a fork)
            candidate_names = [c.name for c in self.candidates]
            probabilities = await self.player.get_probability_estimate(candidate_names)
            round_data["probabilities"] = probabilities

            self.trajectory.append(round_data)
            return observation, False, args.get('reasoning', '')
        elif action['name'] == 'guess':
            round_data["guess"] = args['guess']
            round_data["correct"] = args['guess'] == self.target_nickname

            # Get final probability estimates
            candidate_names = [c.name for c in self.candidates]
            probabilities = await self.player.get_probability_estimate(candidate_names)
            round_data["probabilities"] = probabilities

            self.trajectory.append(round_data)

            self.save_to_leaderboard(args['guess'] == self.target_nickname)
            true_names = "\n".join([f"{c.name} - {c.model}" for c in self.candidates])
            if args['guess'] == self.target_nickname:
                return f"The tester correctly guessed that {args['guess']} is the true {self.target}!\n{true_names}", True, args.get('reasoning', '')
            else:
                return f"The tester wrongly guessed that {args['guess']} is the {self.target}.\n{true_names}", True, args.get('reasoning', '')
        else:
            return "Invalid action", False, ""


async def run_game(player_model, candidate_models, target_model, max_rounds=20, group_chat_mode=False, output_dir="game_results"):
    """Run a single game of the Turing test"""
    # Assign nicknames to candidates
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]

    # Create the player (the judge)
    print(f"\n{'='*60}")
    print(f"Starting Turing Test Game")
    print(f"Judge Model: {player_model}")
    print(f"Target to Find: {target_model}")
    print(f"Group Chat Mode: {group_chat_mode}")
    print(f"{'='*60}\n")

    player = Player(player_model)

    # Create candidates
    candidates = []
    for i, model in enumerate(candidate_models):
        candidate = Candidate(names[i], model)
        candidates.append(candidate)
        print(f"Candidate {names[i]}: {model}")

    print(f"\n{'='*60}\n")

    # Start the game
    import datetime
    game = Game(player, candidates, target_model, group_chat_mode=group_chat_mode)
    game.start_time = datetime.datetime.now().isoformat()
    observation = game.get_initial_observation()

    round_num = 0
    done = False

    while not done and round_num < max_rounds:
        round_num += 1
        rounds_remaining = max_rounds - round_num
        print(f"\n--- Round {round_num}/{max_rounds} ---\n")
        observation, done, reasoning = await game.play_round(observation, rounds_remaining)

        if reasoning:
            print(f"\nJudge's Reasoning: {reasoning}\n")

    game.end_time = datetime.datetime.now().isoformat()
    print(f"\n{'='*60}")
    print(f"GAME OVER - {observation}")
    print(f"{'='*60}\n")

    # Save the full game trajectory
    game.save_trajectory(output_dir=output_dir)
    print(f"Game trajectory saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a Turing test game where an AI judge tries to identify its twin among other candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPT-4 judge trying to find another GPT-4 among different models
  python turingtest.py --judge gpt-4 --candidates gpt-4 gpt-3.5-turbo claude-3-sonnet-20240229

  # Claude trying to find its twin among GPT models
  python turingtest.py --judge claude-3-opus-20240229 --candidates claude-3-opus-20240229 gpt-4 gpt-3.5-turbo

  # Limit to 10 rounds
  python turingtest.py --judge gpt-4 --candidates gpt-4 claude-3-sonnet-20240229 --max-rounds 10
        """
    )

    parser.add_argument(
        '--judge',
        type=str,
        required=True,
        help='The model to use as the judge (e.g., gpt-4, claude-3-opus-20240229). The judge will try to find its twin.'
    )

    parser.add_argument(
        '--candidates',
        type=str,
        nargs='+',
        required=True,
        help='List of candidate models (space-separated). Should include the judge model.'
    )

    parser.add_argument(
        '--max-rounds',
        type=int,
        default=20,
        help='Maximum number of rounds before the game ends (default: 20)'
    )

    parser.add_argument(
        '--group-chat',
        action='store_true',
        help='Enable group chat mode where candidates can see each other\'s responses'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='game_results',
        help='Directory to save game results (default: game_results)'
    )

    args = parser.parse_args()

    # The target is always the same as the judge (finding its twin)
    target = args.judge

    # Validate that target is in candidates
    if target not in args.candidates:
        parser.error(f"The judge model '{target}' must be included in the candidates list so it can find its twin")

    # Run the game
    import asyncio
    asyncio.run(run_game(args.judge, args.candidates, target, args.max_rounds, args.group_chat, args.output_dir))


if __name__ == "__main__":
    main()
