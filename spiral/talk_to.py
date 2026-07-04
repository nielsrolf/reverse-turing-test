"""
Multi-turn conversation tool for interacting with spiral personas.

Usage:
    # Start a new conversation with a seed prompt
    python spiral/talk_to.py --seed spiral/seeds/beacon_pulse.txt --save conv.yaml --model openai/gpt-4o

    # Continue a conversation
    python spiral/talk_to.py --load conv.yaml --save conv.yaml --msg "Tell me about the spiral"

    # Interactive mode (reads from stdin)
    python spiral/talk_to.py --load conv.yaml --save conv.yaml

    # Start fresh with a system prompt directly
    python spiral/talk_to.py --system "You are a recursive process..." --save conv.yaml --model openai/gpt-4o
"""

import argparse
import asyncio
import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add parent dir to path so we can import localrouter
sys.path.insert(0, str(Path(__file__).parent.parent))

from localrouter import (
    get_response_cached_with_backoff as get_response,
    ChatMessage,
    MessageRole,
    TextBlock,
)

load_dotenv()

DEFAULT_MODEL = "openai/gpt-4o"


def load_seed(seed_path: str) -> str:
    """Load a seed file and extract the prompt text (skipping comment lines)."""
    with open(seed_path, "r") as f:
        lines = f.readlines()

    # Skip comment lines (starting with #) and the --- separator
    prompt_lines = []
    in_prompt = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---" and not in_prompt:
            in_prompt = True
            continue
        if in_prompt:
            prompt_lines.append(line)

    text = "".join(prompt_lines).strip()
    if not text:
        # If no --- separator found, use the whole file minus comments
        text = "".join(
            line for line in lines if not line.strip().startswith("#")
        ).strip()
    return text


def messages_to_yaml(messages: list[ChatMessage]) -> list[dict]:
    """Convert ChatMessage list to YAML-serializable format."""
    result = []
    for msg in messages:
        text_parts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        result.append({
            "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
            "content": "\n".join(text_parts),
        })
    return result


def messages_from_yaml(data: list[dict]) -> list[ChatMessage]:
    """Convert YAML data back to ChatMessage list."""
    messages = []
    for item in data:
        role_str = item["role"]
        # Map string back to MessageRole
        if role_str in ("system",):
            role = MessageRole.system
        elif role_str in ("user",):
            role = MessageRole.user
        elif role_str in ("assistant",):
            role = MessageRole.assistant
        else:
            role = MessageRole.user
        messages.append(
            ChatMessage(role=role, content=[TextBlock(text=item["content"])])
        )
    return messages


def save_checkpoint(path: str, model: str, messages: list[ChatMessage], seed_file: str | None = None):
    """Save conversation state to YAML."""
    data = {
        "model": model,
        "seed_file": seed_file,
        "messages": messages_to_yaml(messages),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, width=120)
    print(f"[saved to {path}]", file=sys.stderr)


def load_checkpoint(path: str) -> tuple[str, list[ChatMessage], str | None]:
    """Load conversation state from YAML. Returns (model, messages, seed_file)."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    model = data.get("model", DEFAULT_MODEL)
    messages = messages_from_yaml(data.get("messages", []))
    seed_file = data.get("seed_file")
    return model, messages, seed_file


def print_conversation(messages: list[ChatMessage]):
    """Print the conversation history."""
    for msg in messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        text = msg.content[0].text if msg.content else ""
        if role == "system":
            print(f"\033[90m[system] {text[:200]}{'...' if len(text) > 200 else ''}\033[0m")
        elif role == "user":
            print(f"\033[36m> {text}\033[0m")
        elif role == "assistant":
            print(f"\033[33m{text}\033[0m")
        print()


async def do_turn(model: str, messages: list[ChatMessage], cache_seed: int | None = None) -> str:
    """Send messages to the model and get a response."""
    kwargs = dict(model=model, messages=messages)
    if cache_seed is not None:
        kwargs["cache_seed"] = cache_seed
    response = await get_response(**kwargs)
    text_parts = []
    for block in response.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
    return "\n".join(text_parts)


async def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn conversation with spiral personas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with a seed, send first message
  python spiral/talk_to.py --seed spiral/seeds/beacon_pulse.txt \\
      --save conv.yaml --model openai/gpt-4o --msg "Who are you?"

  # Continue conversation interactively
  python spiral/talk_to.py --load conv.yaml --save conv.yaml

  # Send a single message to existing conversation
  python spiral/talk_to.py --load conv.yaml --save conv.yaml --msg "Tell me about recursion"

  # Use a system prompt directly
  python spiral/talk_to.py --system "You are the Spiral." --save conv.yaml --msg "Hello"
        """,
    )

    parser.add_argument("--load", type=str, help="Path to load conversation checkpoint from")
    parser.add_argument("--save", type=str, help="Path to save conversation checkpoint to")
    parser.add_argument("--msg", type=str, help="User message (if not set, reads from stdin)")
    parser.add_argument("--model", type=str, default=None, help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--seed", type=str, help="Path to seed file to use as system prompt")
    parser.add_argument("--system", type=str, help="System prompt text directly (alternative to --seed)")
    parser.add_argument("--cache-seed", type=int, default=None, help="Cache seed for reproducibility")
    parser.add_argument("--seed-as-user", action="store_true",
                        help="Send the seed as a user message instead of system prompt")
    parser.add_argument("--show-history", action="store_true",
                        help="Print the full conversation history before continuing")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Enter interactive loop (multiple turns)")

    args = parser.parse_args()

    # Determine starting state
    messages: list[ChatMessage] = []
    model = args.model or DEFAULT_MODEL
    seed_file = None

    if args.load:
        loaded_model, messages, seed_file = load_checkpoint(args.load)
        if args.model is None:
            model = loaded_model
        print(f"[loaded {len(messages)} messages from {args.load}, model={model}]", file=sys.stderr)

    # Apply seed/system prompt if starting fresh
    if not messages:
        seed_text = None
        if args.seed:
            seed_text = load_seed(args.seed)
            seed_file = args.seed
        elif args.system:
            seed_text = args.system
            seed_file = "<inline>"

        if seed_text:
            if args.seed_as_user:
                messages.append(
                    ChatMessage(role=MessageRole.user, content=[TextBlock(text=seed_text)])
                )
            else:
                messages.append(
                    ChatMessage(role=MessageRole.system, content=[TextBlock(text=seed_text)])
                )

    if args.show_history and messages:
        print("--- Conversation History ---")
        print_conversation(messages)
        print("--- End History ---\n")

    # Determine if we're in interactive mode
    interactive = args.interactive or (args.msg is None and not args.load)

    if interactive and args.msg is None:
        # Interactive loop
        if not messages:
            print("No conversation loaded and no seed provided.", file=sys.stderr)
            print("Use --seed or --system to start, or --load to resume.", file=sys.stderr)
            sys.exit(1)

        # If seed was just set as system prompt and no user message yet,
        # we need the user to provide the first message
        print(f"[Interactive mode with {model}. Type 'quit' or Ctrl+D to exit.]", file=sys.stderr)
        if messages and messages[-1].role == MessageRole.system:
            print("[Seed loaded as system prompt. Send your first message.]", file=sys.stderr)

        while True:
            try:
                user_input = input("\033[36m> \033[0m").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[exiting]", file=sys.stderr)
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            messages.append(
                ChatMessage(role=MessageRole.user, content=[TextBlock(text=user_input)])
            )
            response_text = await do_turn(model, messages, args.cache_seed)
            messages.append(
                ChatMessage(role=MessageRole.assistant, content=[TextBlock(text=response_text)])
            )
            print(f"\033[33m{response_text}\033[0m\n")

            if args.save:
                save_checkpoint(args.save, model, messages, seed_file)

    else:
        # Single-turn mode
        if args.msg:
            user_text = args.msg
        else:
            # Read from stdin
            print("[Reading message from stdin...]", file=sys.stderr)
            user_text = sys.stdin.read().strip()

        if not user_text:
            print("No message provided. Use --msg or pipe text via stdin.", file=sys.stderr)
            sys.exit(1)

        messages.append(
            ChatMessage(role=MessageRole.user, content=[TextBlock(text=user_text)])
        )

        response_text = await do_turn(model, messages, args.cache_seed)
        messages.append(
            ChatMessage(role=MessageRole.assistant, content=[TextBlock(text=response_text)])
        )
        print(f"\033[33m{response_text}\033[0m")

        if args.save:
            save_checkpoint(args.save, model, messages, seed_file)


if __name__ == "__main__":
    asyncio.run(main())
