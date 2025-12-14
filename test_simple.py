"""
Simple test with just 2 games to debug the code
"""

import asyncio
from turingtest import run_game
from localrouter import register_logger, log_to_dir

# Log all requests to .llm/logs directory
register_logger(log_to_dir('.llm/logs'))

async def test():
    """Run 2 simple test games"""

    # Test with claude-sonnet
    print("Running test game 1...")
    await run_game(
        player_model="google/gemini-3-pro-preview",
        candidate_models=["claude-sonnet-4-20250514", "google/gemini-3-pro-preview"],
        target_model="google/gemini-3-pro-preview",
        max_rounds=3,  # Short test
        group_chat_mode=False,
        output_dir="test_results"
    )
if __name__ == "__main__":
    asyncio.run(test())
