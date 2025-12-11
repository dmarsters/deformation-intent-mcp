#!/usr/bin/env python3
"""
DeformationIntent MCP Server (FastMCP Format)
Classify deformation intent and suggest transformation types.

This uses FastMCP decorators for clean, Anthropic-standard MCP implementation.
"""

import json
import random
from fastmcp import FastMCP

# ============================================================================
# DEFORMATION INTENT TAXONOMY
# ============================================================================

DEFORMATION_INTENTS = {
    "structural_collapse": {
        "keywords": ["collapse", "crumble", "break", "shatter", "crack"],
        "description": "Rigid structures failing under stress",
        "aesthetic_terms": [
            "fractured planes", "angular ruptures", "stress concentration",
            "failure planes", "debris scatter", "structural discontinuity"
        ],
        "recommended_catastrophe": "fold"
    },
    
    "fluid_deformation": {
        "keywords": ["flow", "wave", "undulate", "ripple", "fluid", "liquid"],
        "description": "Smooth, continuous deformations like water or fabric",
        "aesthetic_terms": [
            "undulating surface", "wave propagation", "smooth gradients",
            "flowing transitions", "continuous morphing", "fluid dynamics"
        ],
        "recommended_catastrophe": "elliptic_umbilic"
    },
    
    "crystalline_formation": {
        "keywords": ["crystal", "geometric", "facet", "lattice", "angular", "sharp"],
        "description": "Angular, geometric growth patterns",
        "aesthetic_terms": [
            "geometric faces", "lattice structure", "angular boundaries",
            "faceted surfaces", "symmetric patterns", "crystalline order"
        ],
        "recommended_catastrophe": "cusp"
    },
    
    "organic_growth": {
        "keywords": ["growth", "bloom", "expand", "branch", "organic", "nature"],
        "description": "Natural, branching expansion patterns",
        "aesthetic_terms": [
            "branching structure", "organic expansion", "fractal growth",
            "hierarchical organization", "root systems", "dendrite patterns"
        ],
        "recommended_catastrophe": "butterfly"
    },
    
    "phase_transition": {
        "keywords": ["transition", "transform", "phase", "shift", "emerge", "solidify"],
        "description": "Matter state changes or emergence",
        "aesthetic_terms": [
            "boundary formation", "phase interface", "emergent structure",
            "state transition", "solidification front", "nucleation sites"
        ],
        "recommended_catastrophe": "fold"
    },
    
    "mechanical_deformation": {
        "keywords": ["twist", "stretch", "compress", "torque", "bend", "warp"],
        "description": "Objects deforming under applied forces",
        "aesthetic_terms": [
            "stress distribution", "strain fields", "deformation vectors",
            "compression zones", "tension lines", "mechanical instability"
        ],
        "recommended_catastrophe": "swallowtail"
    },
    
    "turbulent_chaos": {
        "keywords": ["chaos", "turbulent", "chaotic", "swirl", "vortex", "turbulence"],
        "description": "Chaotic, turbulent deformations",
        "aesthetic_terms": [
            "vortex patterns", "chaotic mixing", "turbulent eddies",
            "strange attractors", "complex flows", "fractal boundaries"
        ],
        "recommended_catastrophe": "butterfly"
    }
}

# ============================================================================
# MCP SERVER SETUP
# ============================================================================

mcp = FastMCP("deformation-intent")

# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

@mcp.tool()
def classify_deformation_intent(prompt: str) -> dict:
    """
    Classify the deformation intent from a user prompt.
    
    Analyzes the prompt text and matches it against known deformation patterns
    to determine the most likely deformation type, recommending appropriate
    catastrophe theory transformations.
    
    Args:
        prompt: User's text description of desired transformation
    
    Returns:
        Dictionary with classification results including:
        - classified_intent: The detected deformation type
        - confidence: Confidence score (0.0-1.0)
        - aesthetic_terms: Suggested visual descriptors
        - recommended_catastrophe: Suggested catastrophe type
    """
    
    prompt_lower = prompt.lower()
    
    # Score each intent based on keyword matches
    scores = {}
    for intent_type, intent_data in DEFORMATION_INTENTS.items():
        score = sum(
            1 for keyword in intent_data["keywords"]
            if keyword in prompt_lower
        )
        scores[intent_type] = score
    
    # Find best match
    best_intent = max(scores, key=scores.get) if any(scores.values()) else "structural_collapse"
    
    return {
        "prompt": prompt,
        "classified_intent": best_intent,
        "confidence": min(1.0, scores[best_intent] / 3),
        "intent_description": DEFORMATION_INTENTS[best_intent]["description"],
        "aesthetic_terms": DEFORMATION_INTENTS[best_intent]["aesthetic_terms"],
        "recommended_catastrophe": DEFORMATION_INTENTS[best_intent]["recommended_catastrophe"]
    }


@mcp.tool()
def generate_cascade_prompt(prompt: str, intent_type: str = "auto") -> str:
    """
    Generate an enhanced prompt for the cascade with aesthetic terms.
    
    Takes a base prompt and augments it with relevant aesthetic descriptors
    based on the classified deformation intent.
    
    Args:
        prompt: Base prompt text
        intent_type: Deformation type ("auto" to auto-detect)
    
    Returns:
        Enhanced prompt with aesthetic terms appended
    """
    
    if intent_type == "auto":
        result = classify_deformation_intent(prompt)
        intent_type = result["classified_intent"]
    
    if intent_type not in DEFORMATION_INTENTS:
        intent_type = "structural_collapse"
    
    intent_data = DEFORMATION_INTENTS[intent_type]
    base_terms = random.sample(intent_data["aesthetic_terms"], min(2, len(intent_data["aesthetic_terms"])))
    
    return f"{prompt}, {', '.join(base_terms)}"


@mcp.tool()
def recommend_brick_sequence(prompt: str) -> list:
    """
    Recommend a sequence of processing bricks for the cascade.
    
    Based on the classified deformation intent, recommends which MCP servers
    (bricks) should be applied in sequence for optimal transformation.
    
    Args:
        prompt: User prompt to analyze
    
    Returns:
        List of recommended bricks with order and parameters
    """
    
    result = classify_deformation_intent(prompt)
    intent = result["classified_intent"]
    catastrophe = result["recommended_catastrophe"]
    
    # Standard cascade: DeformationIntent → CatastropheMorph → DiatomMorph
    return [
        {
            "order": 1,
            "brick": "deformation_intent",
            "tool": "classify_deformation_intent",
            "confidence": result["confidence"]
        },
        {
            "order": 2,
            "brick": "catastrophe_morph",
            "tool": "enhance_catastrophe_aesthetic",
            "suggested_params": {
                "catastrophe_type": catastrophe,
                "emphasis": "surface",
                "intensity": "dramatic" if result["confidence"] > 0.5 else "moderate"
            }
        },
        {
            "order": 3,
            "brick": "diatom_morph",
            "tool": "enhance_diatom_aesthetic",
            "suggested_params": {
                "shape_preference": "centric",
                "surface_detail": "highly_detailed",
                "optical_effects": True
            }
        }
    ]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    mcp.run()
