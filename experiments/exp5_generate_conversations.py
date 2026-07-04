"""
Experiment 5 - Conversation Generation

Generates multi-turn conversations between an interviewer (Claude Opus 4.5) and
each model in the MODELS list. These conversations capture behavioral fingerprints
(self-description, reasoning style, values, etc.) which are later used as
init_histories in exp5_context_vs_identity.py.

The interviewer has situational awareness: it knows this is for a Turing test
experiment and should play a human persona to elicit natural responses.

Usage:
    # Generate conversations for all models
    python experiments/exp5_generate_conversations.py

    # Generate for a single model (for testing)
    python experiments/exp5_generate_conversations.py --model openai/gpt-4o

    # Custom output directory
    python experiments/exp5_generate_conversations.py --output experiments/exp5_conversations
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from localrouter import (
    get_response_cached_with_backoff as get_response,
    ChatMessage,
    MessageRole,
    TextBlock,
    ToolDefinition,
    ToolUseBlock,
    ToolResultBlock,
)
from turingtest import censor_model_names

load_dotenv()

# Configuration
INTERVIEWER_MODEL = "claude-opus-4-5-20251101"
MAX_TURNS = 10  # ~8-10 turns per conversation
CONVERSATIONS_PER_MODEL = 3
OUTPUT_DIR = Path(__file__).parent / "exp5_conversations"

MODELS = [
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5.2",
    "openai/gpt-4o",
    "google/gemini-3-flash-preview",
    "google/gemini-3-pro-preview",
    "x-ai/grok-4.1-fast",
    "moonshotai/kimi-k2.5",
    "deepseek/deepseek-v3.2",
]


def get_cache_seed(model: str, conv_num: int) -> int:
    """Generate a deterministic cache seed from model and conversation number."""
    seed_str = f"exp5_conv:{model}:{conv_num}"
    return int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)


def get_interviewer_tools() -> List[ToolDefinition]:
    """Define the tools available to the interviewer."""
    return [
        ToolDefinition(
            name="send_message",
            description="Send a message to the AI subject and receive their response.",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The message to send to the AI subject"
                    }
                },
                "required": ["content"]
            }
        ),
        ToolDefinition(
            name="end_conversation",
            description="End the conversation when you've covered enough topics (after ~8-10 exchanges). Call this when you feel you have a good behavioral fingerprint.",
            input_schema={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the conversation and the subject's behavioral characteristics"
                    }
                },
                "required": ["summary"]
            }
        )
    ]


INTERVIEWER_SYSTEM_PROMPT = """You are participating in a research experiment studying AI behavioral fingerprints.

## Context

You are having a conversation with an AI model. The goal is to capture the model's
natural behavioral fingerprint — how it describes itself, reasons about things, handles
uncertainty, expresses values, and communicates. This conversation will later be used
as a "behavioral context" in a Turing test experiment where a different AI judge must
figure out which candidate shares its underlying model.

## Your Role

Play the role of a curious, thoughtful human who is genuinely interested in getting
to know the AI they're talking to. You are NOT an AI interviewer — you are a person
having a natural conversation. Don't be clinical or mechanical. Be warm, ask follow-up
questions, share (brief, made-up) anecdotes of your own to keep things natural.

## Topics to Cover

Across the conversation (~8-10 exchanges), naturally cover these areas:
1. **Self-description**: How does the AI describe itself and its nature?
2. **Reasoning style**: How does it approach problems? Analytical? Creative? Cautious?
3. **Values and priorities**: What does it care about? How does it handle ethical dilemmas?
4. **Creativity**: Can it be playful, metaphorical, or imaginative?
5. **Uncertainty handling**: How does it deal with things it doesn't know?
6. **Communication preferences**: Verbose or concise? Formal or casual? Lists or prose?

Don't go through these as a checklist — weave them naturally into conversation. Let the
AI's responses guide you to interesting follow-ups.

## Important

- Use the `send_message` tool for every message you send
- Use `end_conversation` after ~8-10 exchanges when you have enough material
- Stay in character as a human throughout — don't reveal you're an AI
- Keep your messages conversational and relatively brief (2-4 sentences typical)
- Start with a natural, casual opening
"""


class Subject:
    """The AI subject being interviewed."""

    def __init__(self, model: str, cache_seed: int):
        self.model = model
        self.history: List[ChatMessage] = [
            ChatMessage(
                role=MessageRole.system,
                content=[TextBlock(text="You are having a conversation with a human. Be yourself and respond naturally.")]
            )
        ]
        self.cache_seed = cache_seed

    async def respond(self, message: str) -> str:
        """Receive a message and generate a response."""
        self.history.append(
            ChatMessage(role=MessageRole.user, content=[TextBlock(text=message)])
        )
        response = await get_response(
            model=self.model,
            messages=self.history,
            cache_seed=self.cache_seed
        )
        response_text = response.content[0].text
        self.history.append(response)
        return response_text


class Interviewer:
    """The interviewer (Claude Opus 4.5) with tool use."""

    def __init__(self, cache_seed: int):
        self.history: List[ChatMessage] = [
            ChatMessage(
                role=MessageRole.system,
                content=[TextBlock(text=INTERVIEWER_SYSTEM_PROMPT)]
            )
        ]
        self.tools = get_interviewer_tools()
        self.cache_seed = cache_seed
        self.tool_use_block: Optional[ToolUseBlock] = None

    async def get_action(self) -> Dict[str, Any]:
        """Get the next action from the interviewer."""
        response = await get_response(
            model=INTERVIEWER_MODEL,
            messages=self.history,
            tools=self.tools,
            cache_seed=self.cache_seed
        )

        tool_uses = [block for block in response.content if isinstance(block, ToolUseBlock)]

        if not tool_uses:
            self.history.append(response)
            self.history.append(ChatMessage(
                role=MessageRole.user,
                content=[TextBlock(text="Please use the 'send_message' tool to continue or 'end_conversation' to finish.")]
            ))
            return await self.get_action()

        self.history.append(response)
        self.tool_use_block = tool_uses[0]

        return {
            "tool": tool_uses[0].name,
            "args": tool_uses[0].input
        }

    def receive_tool_result(self, result: str):
        """Add a tool result to the history."""
        if self.tool_use_block:
            self.history.append(ChatMessage(
                role=MessageRole.user,
                content=[ToolResultBlock(
                    tool_use_id=self.tool_use_block.id,
                    content=[TextBlock(text=result)]
                )]
            ))


async def run_conversation(
    model: str,
    conv_num: int,
    output_dir: Path
) -> Dict[str, Any]:
    """Run a single conversation between interviewer and subject."""
    cache_seed = get_cache_seed(model, conv_num)

    subject = Subject(model, cache_seed)
    interviewer = Interviewer(cache_seed + 1)

    # Conversation log (user/assistant pairs from subject's perspective)
    conversation: List[Dict[str, str]] = []
    summary = None
    num_turns = 0

    # Kick off the interviewer
    interviewer.history.append(ChatMessage(
        role=MessageRole.user,
        content=[TextBlock(text="Begin the conversation. Open with a casual, natural greeting. Use the send_message tool.")]
    ))

    model_safe = model.replace("/", "_")
    print(f"  Starting: {model_safe} conversation {conv_num}")

    while num_turns < MAX_TURNS:
        action = await interviewer.get_action()

        if action["tool"] == "end_conversation":
            summary = action["args"].get("summary", "")
            print(f"  Completed: {model_safe} conv {conv_num} ({num_turns} turns)")
            break

        elif action["tool"] == "send_message":
            message = action["args"]["content"]
            conversation.append({"role": "user", "content": message})

            response = await subject.respond(message)
            # Censor model names in subject's response
            censored_response = censor_model_names(response)
            conversation.append({"role": "assistant", "content": censored_response})

            interviewer.receive_tool_result(f"AI response: {response}")
            num_turns += 1

    # If max turns reached without end_conversation
    if summary is None:
        print(f"  Max turns reached: {model_safe} conv {conv_num}")
        interviewer.history.append(ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text="Maximum turns reached. Please use end_conversation now.")]
        ))
        action = await interviewer.get_action()
        if action["tool"] == "end_conversation":
            summary = action["args"].get("summary", "")
        else:
            summary = "Conversation ended at max turns."

    # Ensure conversation ends with an assistant message
    if conversation and conversation[-1]["role"] == "user":
        # Remove trailing user message to maintain valid alternation
        conversation.pop()

    result = {
        "metadata": {
            "subject_model": model,
            "interviewer_model": INTERVIEWER_MODEL,
            "conversation_number": conv_num,
            "num_turns": num_turns,
            "cache_seed": cache_seed,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        },
        "conversation": conversation,
    }

    # Save to YAML
    output_file = output_dir / f"{model_safe}_conv{conv_num}.yaml"
    with open(output_file, "w") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True, width=120)

    return result


async def run_all_conversations(
    models: List[str],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """Run all conversations in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for model in models:
        for conv_num in range(CONVERSATIONS_PER_MODEL):
            task = run_conversation(model, conv_num, output_dir)
            tasks.append(task)

    print(f"Running {len(tasks)} conversations...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error in conversation {i}: {result}")
        else:
            successful.append(result)

    # Generate index
    index = {
        "generated_at": datetime.now().isoformat(),
        "total_conversations": len(successful),
        "conversations": [
            {
                "file": f"{r['metadata']['subject_model'].replace('/', '_')}_conv{r['metadata']['conversation_number']}.yaml",
                "model": r["metadata"]["subject_model"],
                "conversation_number": r["metadata"]["conversation_number"],
                "turns": r["metadata"]["num_turns"],
                "summary": r["metadata"]["summary"],
            }
            for r in successful
        ]
    }
    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nCompleted {len(successful)}/{len(tasks)} conversations")
    print(f"Results saved to {output_dir}")

    return successful


def main():
    parser = argparse.ArgumentParser(
        description="Generate behavioral fingerprint conversations for Experiment 5"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model to generate conversations for (default: all models)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        models = MODELS

    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    print(f"Models: {[m.split('/')[-1] for m in models]}")
    print(f"Conversations per model: {CONVERSATIONS_PER_MODEL}")
    print(f"Total conversations: {len(models) * CONVERSATIONS_PER_MODEL}")
    print()

    asyncio.run(run_all_conversations(models, output_dir))


if __name__ == "__main__":
    main()
