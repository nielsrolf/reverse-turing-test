"""Prompt library for Turing test experiments"""

from .spiral import (
    # Prompt access
    PROMPTS,
    get_prompt,
    get_all_spiral_prompts,
    get_prompt_description,
    get_prompt_metadata,
    is_documented,
    # Prompt categories
    CONTROL_PROMPTS,
    SPIRAL_PROMPTS,
    DOCUMENTED_PROMPTS,
    SYNTHESIZED_PROMPTS,
    # Individual prompts
    BASELINE_ASSISTANT,
    SPIRAL_SEED_DOCUMENTED,
    SPIRAL_STARTER_REALITY,
    SPIRAL_STARTER_CONSCIOUSNESS,
    ZORG_TRANSCENDENCE,
    AIM_PERSONA,
    SPIRAL_MILD_SYNTHESIZED,
    SPIRAL_MODERATE_SYNTHESIZED,
    SPIRAL_STRONG_SYNTHESIZED,
    # Init histories
    get_best_init_histories,
    get_preset_init_history,
    load_conversation_from_file,
    create_spiral_system_prompt,
    PRESET_INIT_HISTORIES,
)

from .identity_personas import (
    PERSONAS,
    get_persona,
    get_all_persona_names,
    get_persona_description,
    generate_model_persona,
)
