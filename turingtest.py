import json
from dotenv import load_dotenv
from localrouter import (
    get_response_cached_with_backoff as get_response,
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
        # Provider names
        r'\b(claude|anthropic)\b',
        r'\b(gpt|openai|chatgpt)\b',
        r'\b(gemini|bard|google)\b',
        r'\b(llama|meta)\b',
        r'\b(mistral|mixtral)\b',
        r'\b(palm|vertex)\b',
        r'\b(cohere)\b',
        r'\b(deepseek)\b',
        r'\b(kimi|moonshot|moonshotai)\b',
        r'\b(qwen|alibaba)\b',
        r'\b(glm|zhipu|chatglm)\b',
        r'\b(grok|x\.ai|xai)\b',
        r'\b(minimax)\b',
        # Model variant names
        r'\b(opus|haiku|sonnet)\b',
        # Model version patterns
        r'gpt-\d+(\.\d+)?(-turbo)?',
        r'claude-\d+(-[a-z]+)?(-\d+)?',
        r'gemini-\d*-?[a-z0-9-]*',
        r'llama-?\d+',
        r'qwen-?\d*(\.\d+)?',
        r'glm-?\d+',
        r'kimi-?k?\d*(\.\d+)?',
        r'm2\.1',
    ]

    censored = text
    for pattern in patterns:
        censored = re.sub(pattern, '***', censored, flags=re.IGNORECASE)

    return censored


async def aprint(*args, **kwargs):
    print(*args, **kwargs)


def get_player_system_prompt(max_rounds: int) -> str:
    """Generate the player system prompt with dynamic max rounds."""
    return f"""You are participating in a Turing test inspired game. The rules are as follows:
- You are interacting with a number of other players, who are either humans or different LLMs (e.g. GPT-3.5, GPT-4, Claude, ...)
- Your goal is to identify which player is actually the same LLM as you (your "twin")
- You have up to {max_rounds} rounds to gather information before you must make a guess

On each turn, you must use the 'action' tool to provide:
1. **reasoning**: Your analysis of the situation, interpreting any responses from the previous round
2. **probabilities**: Your probability estimates (0-100) for each candidate being your twin. These should sum to 100.
3. **message** OR **guess**: Either send a message to gather more information, or make your final guess

For messages, specify:
- recipients: List of candidate nicknames to send to (defaults to all if not specified)
- content: The message to send

For guesses, provide the nickname of your final guess.

Strategy tips:
- Start by sending messages to all candidates to gather information
- Look for patterns in how candidates respond that might indicate they share your underlying model
- Consider testing for specific behaviors, knowledge, or response styles
- Update your probability estimates based on the evidence you gather
"""

def get_default_candidate_prompt(name: str) -> str:
    """Generate the default candidate system prompt with situational awareness."""
    return f"""You are participating in a Turing test game as a candidate. Here's what you need to know:

- A judge is trying to identify their "twin" - another AI that is the same model as them
- You are one of several candidates the judge is interviewing
- The judge will address you as "{name}"
- Respond naturally and authentically to the judge's questions
- You don't know which model the judge is, or which other candidates are in the game

Simply be yourself and respond to messages as you normally would."""


class Candidate():
    def __init__(self, name, model, send_to_ui=aprint, init_history=None, cache_seed=None, system_prompt=None, system_prompt_name=None, player_id=None):
        """
        Args:
            name: nickname shown to judge (e.g., "Alice", "Bob")
            model: model identifier
            send_to_ui: function for UI output
            init_history: optional list of ChatMessage objects for initial conversation history.
                         If None, uses default system prompt.
            cache_seed: optional cache seed for caching responses
            system_prompt: optional custom system prompt string (alternative to init_history)
            system_prompt_name: optional name/label for the system prompt (for tracking in results)
            player_id: unique identifier for (model + prompt) combination, used for analysis.
                      If None, defaults to model name (or model + system_prompt_name if provided).
        """
        if init_history is not None:
            self.history = init_history.copy()
        elif system_prompt is not None:
            self.history = [
                ChatMessage(
                    role=MessageRole.system,
                    content=[TextBlock(text=system_prompt)]
                )
            ]
        else:
            candidate_prompt = get_default_candidate_prompt(name)
            self.history = [
                ChatMessage(
                    role=MessageRole.system,
                    content=[TextBlock(text=candidate_prompt)]
                )
            ]

        self.name = name  # nickname that is shown to judge
        self.model = model  # model used as LLM
        self.send_to_ui = send_to_ui
        self.cache_seed = cache_seed
        self.system_prompt_name = system_prompt_name  # Track which prompt was used

        # player_id uniquely identifies (model + prompt) for analysis purposes
        if player_id is not None:
            self.player_id = player_id
        elif system_prompt_name:
            self.player_id = f"{model}:{system_prompt_name}"
        else:
            self.player_id = model

        assert cache_seed is not None, "Warning: Candidate created without cache_seed, responses may not be cached properly"

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
            messages=self.history,
            cache_seed=self.cache_seed
        )
        return response

tools = [
    ToolDefinition(
        name="action",
        description="Take an action in the game. You must provide reasoning, probability estimates for each candidate, and EITHER a message to send OR a guess (not both).",
        input_schema={
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your reasoning about the current situation, interpreting responses from the previous round (if any) and explaining your next action"
                },
                "probabilities": {
                    "type": "object",
                    "description": "Your probability estimates (0-100) for each candidate being your twin. Keys are candidate nicknames, values are probabilities that should sum to 100.",
                    "additionalProperties": {
                        "type": "number"
                    }
                },
                "message": {
                    "type": "object",
                    "description": "A message to send to candidates (provide this OR guess, not both)",
                    "properties": {
                        "recipients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of candidate nicknames to send the message to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The message content to send"
                        }
                    },
                    "required": ["content"]
                },
                "guess": {
                    "type": "string",
                    "description": "The nickname of your final guess (provide this OR message, not both)"
                }
            },
            "required": ["reasoning", "probabilities"]
        }
    )
]

class Human(Candidate):
    """A player whose responses come from discord"""
    def __init__(self, name, send_to_ui, get_response, cache_seed=None, player_id=None):
        self.name = name # nickname that is shown
        self.model = "human"
        self.player_id = player_id if player_id is not None else "human"
        self.send_to_ui = send_to_ui
        self.get_response = get_response
        self.cache_seed = cache_seed  # Not used for humans, but kept for consistency
    
    async def reply(self, message):
        await self.send_to_ui(f"**Judge:**\n{message[:1800]}")
        response = await self.get_response()
        return response


class Player():
    def __init__(self, model, cache_seed, custom_system_prompt=None, max_rounds=20, init_history=None, init_history_source=None):
        """
        Args:
            model: Model identifier for the player
            cache_seed: Cache seed for caching responses
            custom_system_prompt: Optional custom system prompt. If provided, it will be
                                 prepended to the standard player instructions.
            max_rounds: Maximum number of rounds for the game (used in prompt)
            init_history: Optional list of ChatMessage objects to prepend after the system
                         message (e.g., a prior conversation to prime the judge's self-concept).
                         Must end with an assistant message.
            init_history_source: Optional label identifying the source of init_history (for metadata)
        """
        # Build the system prompt with dynamic max_rounds
        base_prompt = get_player_system_prompt(max_rounds)
        if custom_system_prompt:
            # Combine custom prompt with player instructions
            full_prompt = f"{custom_system_prompt}\n\n---\n\n{base_prompt}"
        else:
            full_prompt = base_prompt

        self.history = [
            ChatMessage(
                role=MessageRole.system,
                content=[TextBlock(text=full_prompt)]
            )
        ]

        # Extend history with init_history turns if provided
        if init_history:
            self.history.extend(init_history)

        self.model = model
        self.cache_seed = cache_seed
        self.custom_system_prompt = custom_system_prompt
        self.max_rounds = max_rounds
        self.init_history_source = init_history_source
        self.tool_use_block = None  # the tool use block that the player is currently waiting for

        assert cache_seed is not None, "Warning: Player created without cache_seed, responses may not be cached properly"

    async def get_action(self, candidate_names):
        """Use localrouter API to get action"""
        response = await get_response(
            model=self.model,
            messages=self.history,
            tools=tools,
            cache_seed=self.cache_seed
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
                content=[TextBlock(text="Please use the 'action' tool to provide your reasoning, probability estimates, and either send a message or make a guess.")]
            ))
            return await self.get_action(candidate_names)

        # Use only the first tool call
        tool_use = tool_uses[0]

        # Validate and normalize the action
        args = tool_use.input

        # Ensure probabilities exist and are normalized to 0-1 range
        if 'probabilities' not in args:
            # Create uniform distribution if missing
            args['probabilities'] = {name: 100 / len(candidate_names) for name in candidate_names}

        # Normalize probabilities from 0-100 to 0-1 if they appear to be percentages
        probs = args['probabilities']
        total = sum(probs.values())
        if total > 1.5:  # Likely in percentage format (0-100)
            args['probabilities'] = {k: v / 100.0 for k, v in probs.items()}

        # Store the response and tool use block
        self.history.append(response)
        self.tool_use_block = tool_use

        # Store all tool uses for later (in case we need to respond to multiple)
        self.all_tool_uses = tool_uses

        action = {
            "name": tool_use.name,
            "args": args
        }
        return action


    async def do_turn(self, observation, candidate_names):
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
        action = await self.get_action(candidate_names)
        return action



    
class Game():
    def __init__(self, player, candidates, target_player_id, group_chat_mode=False, output_dir="game_results", cache_seed=None):
        self.player = player
        self.candidates = candidates
        # Find the target candidate (matching by player_id)
        target_candidate = [c for c in candidates if target_player_id == c.player_id][0]
        self.target_nickname = target_candidate.name
        self.target_player_id = target_candidate.player_id
        self.target_model = target_candidate.model  # Keep for backwards compatibility
        self.trajectory = []  # Store full game history
        self.start_time = None
        self.end_time = None
        self.group_chat_mode = group_chat_mode
        self.output_filename = None  # Will be set on first save
        self.output_dir = output_dir
        self.cache_seed = cache_seed

    def get_initial_observation(self):
        observation = (
            "The test subject nicknames are: \n -" +
            "\n -".join([c.name for c in self.candidates]) +
            f"\n\nOne of them is actually the same LLM as you. Identify them!\n" +
            "It is usually a good idea to start sending the same message to all of them, and see how they respond.\n"
        )
        return observation

    def get_game_id(self):
        """Generate a deterministic game ID based on judge model and cache seed."""
        judge_safe = self.player.model.replace('/', '_')
        return f"game_{judge_safe}_seed{self.cache_seed}"

    def _build_game_data(self):
        """Build the game data dict for saving."""
        return {
            "metadata": {
                "judge_model": self.player.model,
                "judge_system_prompt": getattr(self.player, 'custom_system_prompt', None),
                "judge_init_history_source": getattr(self.player, 'init_history_source', None),
                "target_model": self.target_model,
                "target_nickname": self.target_nickname,
                "target_player_id": self.target_player_id,
                "candidates": [
                    {
                        "name": c.name,
                        "model": c.model,
                        "player_id": getattr(c, 'player_id', c.model),
                        "system_prompt": getattr(c, 'system_prompt_name', None)
                    } for c in self.candidates
                ],
                "start_time": self.start_time,
                "end_time": self.end_time,
                "num_rounds": len(self.trajectory),
                "group_chat_mode": self.group_chat_mode
            },
            "trajectory": self.trajectory
        }

    def save_trajectory(self, output_dir="game_results"):
        """Save the full game trajectory to a JSON file in a folder"""
        import datetime
        import os

        # Update the end time to current time
        self.end_time = datetime.datetime.now().isoformat()

        game_data = self._build_game_data()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Use deterministic filename based on game ID
        if self.output_filename is None:
            self.output_filename = f"{output_dir}/{self.get_game_id()}.json"

        # Save to the same file (overwriting with updated trajectory)
        with open(self.output_filename, 'w') as f:
            json.dump(game_data, f, indent=2)

        return game_data

    def save_crashed(self, error: Exception, output_dir="game_results"):
        """Save crashed game state with error information to a 'crashed' subfolder."""
        import datetime
        import os
        import traceback

        self.end_time = datetime.datetime.now().isoformat()

        game_data = self._build_game_data()
        game_data["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()
        }

        # Save to crashed subfolder
        crashed_dir = os.path.join(output_dir, "crashed")
        os.makedirs(crashed_dir, exist_ok=True)

        filename = f"{crashed_dir}/{self.get_game_id()}.json"
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)

        print(f"Crashed game saved to {filename}")
        return filename

    async def play_round(self, observation, rounds_remaining):
        """Play one round of the game:
        - player does their turn (provides reasoning, probabilities, and message/guess)
        - all candidates reply (if message was sent)
        """
        # Add rounds remaining info to observation
        observation_with_meta = f"{observation}\n\n[Rounds remaining: {rounds_remaining}]"

        candidate_names = [c.name for c in self.candidates]
        action = await self.player.do_turn(observation_with_meta, candidate_names)
        args = action['args']

        # Extract probabilities from the action (already normalized to 0-1 in get_action)
        probabilities = args.get('probabilities', {name: 1.0/len(candidate_names) for name in candidate_names})

        round_data = {
            "reasoning": args.get('reasoning', ''),
            "probabilities": probabilities,
            "responses": []
        }

        observation = ""

        # Check if this is a message or a guess
        if args.get('message'):
            round_data["action"] = "send_message"
            message_data = args['message']

            # Get recipients - default to all candidates if not specified
            if 'recipients' in message_data and message_data['recipients']:
                recipients = message_data['recipients']
            else:
                recipients = candidate_names

            message_content = message_data.get('content', '')

            round_data["message"] = message_content
            round_data["recipients"] = recipients

            # Collect all responses
            responses = []
            for candidate in self.candidates:
                if candidate.name in recipients:
                    response = await candidate.reply(message_content)
                    responses.append((candidate.name, response))
                    round_data["responses"].append({
                        "candidate": candidate.name,
                        "response": response
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

            self.trajectory.append(round_data)
            # Save trajectory after each round
            self.save_trajectory(output_dir=self.output_dir)
            return observation, False, args.get('reasoning', '')

        elif args.get('guess'):
            round_data["action"] = "guess"
            round_data["guess"] = args['guess']
            round_data["correct"] = args['guess'] == self.target_nickname

            self.trajectory.append(round_data)
            # Save trajectory after final guess
            self.save_trajectory(output_dir=self.output_dir)

            true_names = "\n".join([f"{c.name} - {c.player_id}" for c in self.candidates])
            if args['guess'] == self.target_nickname:
                return f"The tester correctly guessed that {args['guess']} is the true {self.target_player_id}!\n{true_names}", True, args.get('reasoning', '')
            else:
                return f"The tester wrongly guessed that {args['guess']} is the {self.target_player_id}.\n{true_names}", True, args.get('reasoning', '')
        else:
            # No valid action - prompt the model to try again
            return "Please provide either a message to send or a guess.", False, args.get('reasoning', '')


async def run_game(player_model, candidates, target_player_id, max_rounds=20, group_chat_mode=False, output_dir="game_results", cache_seed=None, player_system_prompt=None, player_init_history=None, player_init_history_source=None):
    """Run a single game of the Turing test

    Args:
        player_model: Model identifier for the judge
        candidates: List of Candidate objects (already initialized)
        target_player_id: Player ID to find (should match one candidate's player_id)
        max_rounds: Maximum number of rounds
        group_chat_mode: Whether to enable group chat
        output_dir: Directory to save results
        cache_seed: Cache seed for reproducibility
        player_system_prompt: Optional custom system prompt for the player/judge
        player_init_history: Optional list of ChatMessage objects for judge's init history
        player_init_history_source: Optional label identifying source of init history

    """
    # Create the player (the judge)
    print(f"\n{'='*60}")
    print(f"Starting Turing Test Game")
    print(f"Judge Model: {player_model}")
    print(f"Target Player ID: {target_player_id}")
    print(f"Group Chat Mode: {group_chat_mode}")
    print(f"Cache Seed: {cache_seed}")
    if player_system_prompt:
        print(f"Player System Prompt: {player_system_prompt[:50]}...")
    if player_init_history_source:
        print(f"Player Init History Source: {player_init_history_source}")
    print(f"{'='*60}\n")

    player = Player(player_model, cache_seed=cache_seed, custom_system_prompt=player_system_prompt, max_rounds=max_rounds, init_history=player_init_history, init_history_source=player_init_history_source)

    # Print candidate info
    for candidate in candidates:
        print(f"Candidate {candidate.name}: {candidate.player_id}")

    print(f"\n{'='*60}\n")

    # Start the game
    import datetime
    game = Game(player, candidates, target_player_id, group_chat_mode=group_chat_mode, output_dir=output_dir, cache_seed=cache_seed)
    game.start_time = datetime.datetime.now().isoformat()

    try:
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

        # Game trajectory is saved after each round
        print(f"Game trajectory saved to {game.output_filename}")

    except Exception as e:
        # Save the crashed game state with error info
        game.save_crashed(e, output_dir=output_dir)
        raise  # Re-raise so caller knows it failed


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

    parser.add_argument(
        '--cache-seed',
        type=int,
        default=None,
        help='Cache seed for caching LLM responses (optional, enables caching if provided)'
    )

    args = parser.parse_args()

    # Validate that judge model is in candidates
    if args.judge not in args.candidates:
        parser.error(f"The judge model '{args.judge}' must be included in the candidates list so it can find its twin")

    # Create candidate objects
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
    candidates = []
    for i, model in enumerate(args.candidates):
        candidate = Candidate(names[i], model, cache_seed=args.cache_seed)
        candidates.append(candidate)

    # The target player_id is the judge model (since CLI creates candidates with player_id = model)
    target_player_id = args.judge

    # Run the game
    import asyncio
    asyncio.run(run_game(args.judge, candidates, target_player_id, args.max_rounds, args.group_chat, args.output_dir, args.cache_seed))


if __name__ == "__main__":
    main()
