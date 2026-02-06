#!/usr/bin/env python3
"""
DeformationIntent MCP Server - Lushy Brick Edition
Phase 2.6 Rhythmic Presets + Phase 2.7 Attractor Visualization

Classify deformation intent from natural language and recommend transformation
sequences for the Lushy brick cascade.

Follows the three-layer olog architecture for cost-efficient prompt enhancement.

Architecture:
- Layer 1: Pure taxonomy lookup from YAML olog (zero LLM cost)
- Layer 2: Deterministic classification, mapping, and dynamics (zero LLM cost)
- Layer 3: Returns structured data for Claude synthesis

Phase 2.6 Additions:
- 5D morphospace for deformation intent parameter space
- 7 canonical state coordinates (one per intent type)
- 5 rhythmic presets oscillating between intent states
- Sinusoidal, triangular, and square oscillation patterns

Phase 2.7 Additions:
- Visual vocabulary extraction from parameter coordinates
- Attractor visualization prompt generation
- Composite, split-view, and sequence prompt modes
- Domain-registry-compatible configuration export
"""

from pathlib import Path
from fastmcp import FastMCP
import yaml
import random
import math
from typing import Dict, List, Tuple, Optional

# ============================================================================
# LOAD TAXONOMY FROM OLOG
# ============================================================================

OLOG_PATH = Path(__file__).parent / "ologs" / "deformation_intent.yaml"


def load_olog():
    """Load the deformation intent taxonomy from YAML."""
    with open(OLOG_PATH, "r") as f:
        return yaml.safe_load(f)


# Load at module level for performance
OLOG = load_olog()
DEFORMATION_INTENTS = OLOG["deformation_intents"]
CASCADE_INTEGRATION = OLOG["cascade_integration"]
INTENSITY_PROFILES = OLOG["intensity_profiles"]


# ============================================================================
# PHASE 2.6 - MORPHOSPACE DEFINITION
# ============================================================================
# 5D parameter space capturing the key aesthetic axes of deformation.
# Each parameter is normalized [0.0, 1.0].

PARAMETER_NAMES = [
    "structural_rigidity",   # 0.0 = fluid/amorphous → 1.0 = rigid/crystalline
    "deformation_rate",      # 0.0 = gradual/static → 1.0 = sudden/explosive
    "geometric_order",       # 0.0 = chaotic/organic → 1.0 = geometric/lattice
    "force_continuity",      # 0.0 = discrete/impulse → 1.0 = continuous/flowing
    "scale_complexity",      # 0.0 = uniform/simple → 1.0 = fractal/multi-scale
]

# Canonical state coordinates — one per deformation intent type.
# These define the morphospace vertices that presets oscillate between.

DEFORMATION_COORDS = {
    "structural_collapse": {
        "structural_rigidity": 0.85,
        "deformation_rate": 0.95,
        "geometric_order": 0.40,
        "force_continuity": 0.15,
        "scale_complexity": 0.55,
    },
    "fluid_deformation": {
        "structural_rigidity": 0.10,
        "deformation_rate": 0.30,
        "geometric_order": 0.15,
        "force_continuity": 0.95,
        "scale_complexity": 0.45,
    },
    "crystalline_formation": {
        "structural_rigidity": 0.95,
        "deformation_rate": 0.15,
        "geometric_order": 0.95,
        "force_continuity": 0.80,
        "scale_complexity": 0.25,
    },
    "organic_growth": {
        "structural_rigidity": 0.25,
        "deformation_rate": 0.20,
        "geometric_order": 0.30,
        "force_continuity": 0.85,
        "scale_complexity": 0.90,
    },
    "phase_transition": {
        "structural_rigidity": 0.50,
        "deformation_rate": 0.60,
        "geometric_order": 0.50,
        "force_continuity": 0.50,
        "scale_complexity": 0.60,
    },
    "mechanical_deformation": {
        "structural_rigidity": 0.80,
        "deformation_rate": 0.45,
        "geometric_order": 0.55,
        "force_continuity": 0.75,
        "scale_complexity": 0.30,
    },
    "turbulent_chaos": {
        "structural_rigidity": 0.10,
        "deformation_rate": 0.70,
        "geometric_order": 0.05,
        "force_continuity": 0.40,
        "scale_complexity": 0.95,
    },
}


# ============================================================================
# PHASE 2.6 - RHYTHMIC PRESETS
# ============================================================================
# Each preset defines a periodic oscillation between two intent states,
# creating a rhythmic aesthetic trajectory in 5D parameter space.

PHASE26_PRESETS = {
    "collapse_flow_cycle": {
        "state_a": "structural_collapse",
        "state_b": "fluid_deformation",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 18,
        "description": (
            "Rigid failure dissolving into fluid flow — the moment "
            "architecture becomes liquid. Captures destruction-to-motion."
        ),
    },
    "order_chaos_oscillation": {
        "state_a": "crystalline_formation",
        "state_b": "turbulent_chaos",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 22,
        "description": (
            "Geometric lattice and chaotic turbulence trading dominance. "
            "The boundary where order and entropy negotiate."
        ),
    },
    "growth_collapse_rhythm": {
        "state_a": "organic_growth",
        "state_b": "structural_collapse",
        "pattern": "triangular",
        "num_cycles": 3,
        "steps_per_cycle": 20,
        "description": (
            "Creation and destruction in slow triangular exchange — "
            "life builds what entropy dismantles."
        ),
    },
    "phase_breathing": {
        "state_a": "phase_transition",
        "state_b": "crystalline_formation",
        "pattern": "sinusoidal",
        "num_cycles": 5,
        "steps_per_cycle": 15,
        "description": (
            "Solidification and melting in rhythmic breathing. "
            "Matter oscillates between committed and liminal states."
        ),
    },
    "stress_release_pulse": {
        "state_a": "mechanical_deformation",
        "state_b": "fluid_deformation",
        "pattern": "square",
        "num_cycles": 4,
        "steps_per_cycle": 16,
        "description": (
            "Sharp alternation between stressed rigidity and fluid release — "
            "tension snaps into flow, then locks again."
        ),
    },
}


# ============================================================================
# PHASE 2.7 - VISUAL VOCABULARY FOR PROMPT GENERATION
# ============================================================================
# Each visual type maps a region of the morphospace to image-generation
# keywords. Nearest-neighbour lookup translates any coordinate to prompts.

VISUAL_TYPES = {
    "shattered_architecture": {
        "coords": {
            "structural_rigidity": 0.85,
            "deformation_rate": 0.95,
            "geometric_order": 0.40,
            "force_continuity": 0.15,
            "scale_complexity": 0.55,
        },
        "keywords": [
            "fractured concrete planes",
            "angular debris field",
            "dust-filled collapse",
            "sharp rupture lines",
            "gravitational failure",
            "exposed rebar and stress fractures",
            "monochrome destruction",
        ],
    },
    "liquid_morph": {
        "coords": {
            "structural_rigidity": 0.10,
            "deformation_rate": 0.30,
            "geometric_order": 0.15,
            "force_continuity": 0.95,
            "scale_complexity": 0.45,
        },
        "keywords": [
            "smooth liquid surface",
            "reflective undulating membrane",
            "continuous flowing gradient",
            "translucent depth layers",
            "ripple interference patterns",
            "mercury-like surface tension",
            "slow-motion fluid dynamics",
        ],
    },
    "crystal_lattice": {
        "coords": {
            "structural_rigidity": 0.95,
            "deformation_rate": 0.15,
            "geometric_order": 0.95,
            "force_continuity": 0.80,
            "scale_complexity": 0.25,
        },
        "keywords": [
            "prismatic faceted surfaces",
            "sharp crystalline edges",
            "geometric lattice grid",
            "refractive mineral faces",
            "transparent polyhedral clusters",
            "symmetric angular growth",
            "ice-like precision geometry",
        ],
    },
    "organic_proliferation": {
        "coords": {
            "structural_rigidity": 0.25,
            "deformation_rate": 0.20,
            "geometric_order": 0.30,
            "force_continuity": 0.85,
            "scale_complexity": 0.90,
        },
        "keywords": [
            "branching dendrite networks",
            "fractal root systems",
            "cellular division cascades",
            "hierarchical organic structure",
            "moss-covered growth patterns",
            "living textured surfaces",
            "multi-scale biological detail",
        ],
    },
    "turbulent_vortex": {
        "coords": {
            "structural_rigidity": 0.10,
            "deformation_rate": 0.70,
            "geometric_order": 0.05,
            "force_continuity": 0.40,
            "scale_complexity": 0.95,
        },
        "keywords": [
            "spiralling vortex fields",
            "chaotic smoke dynamics",
            "multi-scale turbulent eddies",
            "fractal energy dissipation",
            "unpredictable swirling volumes",
            "ink-in-water complexity",
            "maelstrom rotational chaos",
        ],
    },
    "stressed_material": {
        "coords": {
            "structural_rigidity": 0.80,
            "deformation_rate": 0.45,
            "geometric_order": 0.55,
            "force_continuity": 0.75,
            "scale_complexity": 0.30,
        },
        "keywords": [
            "visible strain gradients",
            "warped metallic surface",
            "compression deformation lines",
            "tension-stretched material",
            "elastic limit distortion",
            "stressed industrial form",
        ],
    },
    "phase_boundary": {
        "coords": {
            "structural_rigidity": 0.50,
            "deformation_rate": 0.60,
            "geometric_order": 0.50,
            "force_continuity": 0.50,
            "scale_complexity": 0.60,
        },
        "keywords": [
            "state-change interface",
            "solidification front edge",
            "nucleation cluster sites",
            "mixed-phase gradient zone",
            "transformation boundary",
            "liminal matter between states",
        ],
    },
}


# ============================================================================
# INTERNAL HELPER FUNCTIONS
# ============================================================================


def _classify_intent(prompt: str) -> dict:
    """Internal classification logic."""
    prompt_lower = prompt.lower()

    scores = {}
    matched_keywords = {}

    for intent_type, intent_data in DEFORMATION_INTENTS.items():
        matches = [kw for kw in intent_data["keywords"] if kw in prompt_lower]
        score = len(matches)
        if score > 0:
            scores[intent_type] = score
            matched_keywords[intent_type] = matches

    if not scores:
        return {
            "classified_intent": "structural_collapse",
            "confidence": 0.0,
            "confidence_level": "none",
            "reason": "No keywords matched - defaulting to structural_collapse",
            "matched_keywords": [],
            "all_scores": {},
        }

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    confidence = min(1.0, best_score / 3)
    confidence_level = (
        "high" if confidence >= 0.67 else "medium" if confidence >= 0.33 else "low"
    )

    return {
        "classified_intent": best_intent,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "matched_keywords": matched_keywords[best_intent],
        "all_scores": scores,
    }


def _get_cascade_sequence(intent_type: str, confidence: float) -> list:
    """Build cascade sequence recommendation."""
    intent_data = DEFORMATION_INTENTS.get(
        intent_type, DEFORMATION_INTENTS["structural_collapse"]
    )

    intensity = (
        "dramatic"
        if confidence > 0.67
        else "moderate" if confidence > 0.33 else "subtle"
    )

    return [
        {
            "order": 1,
            "brick": "deformation_intent",
            "tool": "classify_deformation_intent",
            "role": "Intent classification and cascade routing",
            "cost": "0 tokens (deterministic)",
        },
        {
            "order": 2,
            "brick": "catastrophe_morph",
            "tool": "enhance_with_catastrophe_aesthetic",
            "role": "Apply mathematical catastrophe topology",
            "suggested_params": {
                "catastrophe_type": intent_data["recommended_catastrophe"],
                "intensity": intensity,
                "emphasis": "surface",
            },
            "cost": "~150-250 tokens (Claude synthesis)",
        },
        {
            "order": 3,
            "brick": "diatom_morph",
            "tool": "build_visualization_prompt",
            "role": "Apply biological silica morphology",
            "suggested_params": {
                "microscope_style": "SEM",
                "structure_name": None,
            },
            "cost": "~100-200 tokens (Claude synthesis)",
        },
    ]


# ============================================================================
# PHASE 2.6 - OSCILLATION GENERATION
# ============================================================================


def _generate_oscillation(num_steps: int, num_cycles: float, pattern: str) -> list:
    """
    Generate oscillation alpha values [0, 1] for trajectory interpolation.

    Args:
        num_steps: Total number of steps in the trajectory.
        num_cycles: Number of full oscillation cycles.
        pattern: One of 'sinusoidal', 'triangular', 'square'.

    Returns:
        List of float alpha values in [0, 1].
    """
    alphas = []
    for i in range(num_steps):
        t = 2.0 * math.pi * num_cycles * i / num_steps
        if pattern == "sinusoidal":
            alphas.append(0.5 * (1.0 + math.sin(t)))
        elif pattern == "triangular":
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alphas.append(2.0 * t_norm if t_norm < 0.5 else 2.0 * (1.0 - t_norm))
        elif pattern == "square":
            t_norm = (t / (2.0 * math.pi)) % 1.0
            alphas.append(0.0 if t_norm < 0.5 else 1.0)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    return alphas


def _generate_preset_trajectory(preset_name: str) -> List[dict]:
    """
    Generate a full Phase 2.6 preset trajectory as a list of state dicts.

    Each step is a dict mapping parameter names to float values, obtained
    by interpolating between state_a and state_b using the oscillation pattern.

    Returns:
        List of state dicts, length = num_cycles * steps_per_cycle.
    """
    config = PHASE26_PRESETS[preset_name]
    state_a = DEFORMATION_COORDS[config["state_a"]]
    state_b = DEFORMATION_COORDS[config["state_b"]]
    total_steps = config["num_cycles"] * config["steps_per_cycle"]

    alphas = _generate_oscillation(total_steps, config["num_cycles"], config["pattern"])

    trajectory = []
    for alpha in alphas:
        state = {}
        for p in PARAMETER_NAMES:
            state[p] = state_a[p] * (1.0 - alpha) + state_b[p] * alpha
        trajectory.append(state)

    return trajectory


# ============================================================================
# PHASE 2.7 - VISUAL VOCABULARY EXTRACTION
# ============================================================================


def _euclidean_distance(a: dict, b: dict) -> float:
    """Euclidean distance between two state dicts over PARAMETER_NAMES."""
    return math.sqrt(sum((a[p] - b[p]) ** 2 for p in PARAMETER_NAMES))


def _extract_visual_vocabulary(state: dict, strength: float = 1.0) -> dict:
    """
    Extract visual vocabulary from a parameter-space coordinate.

    Uses nearest-neighbour lookup against VISUAL_TYPES to find the
    closest canonical visual type, then returns weighted keywords.

    Args:
        state: Dict mapping PARAMETER_NAMES → float values [0, 1].
        strength: Weight multiplier [0, 1]. Controls how many keywords
                  are included (higher = more keywords).

    Returns:
        Dict with nearest_type, distance, keywords, and strength.
    """
    best_type = None
    best_dist = float("inf")

    for vtype_name, vtype_data in VISUAL_TYPES.items():
        dist = _euclidean_distance(state, vtype_data["coords"])
        if dist < best_dist:
            best_dist = dist
            best_type = vtype_name

    vtype = VISUAL_TYPES[best_type]
    # Number of keywords scales with strength
    max_kw = len(vtype["keywords"])
    n_kw = max(2, int(max_kw * min(1.0, strength)))
    selected = vtype["keywords"][:n_kw]

    return {
        "nearest_type": best_type,
        "distance": round(best_dist, 4),
        "keywords": selected,
        "strength": strength,
    }


def _generate_composite_prompt(
    domain_states: Dict[str, dict],
    domain_strengths: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate a single blended image-generation prompt from multi-domain states.

    If only deformation_intent state is provided, generates a single-domain
    prompt. Multiple domains get their keywords interleaved by strength.

    Args:
        domain_states: Mapping domain_id → parameter state dict.
                       Must include 'deformation_intent'.
        domain_strengths: Optional per-domain strength weights [0, 1].

    Returns:
        Composite prompt string ready for image generation.
    """
    if domain_strengths is None:
        domain_strengths = {d: 1.0 for d in domain_states}

    # Collect weighted keyword lists
    all_keywords: List[Tuple[float, str]] = []

    for domain_id, state in domain_states.items():
        strength = domain_strengths.get(domain_id, 1.0)
        if domain_id == "deformation_intent":
            vocab = _extract_visual_vocabulary(state, strength)
            for kw in vocab["keywords"]:
                all_keywords.append((strength, kw))
        # Other domains would be handled here by their own extractors;
        # this server only owns the deformation_intent vocabulary.

    # Sort by strength descending, then join
    all_keywords.sort(key=lambda x: -x[0])
    prompt_parts = [kw for _, kw in all_keywords]

    return ", ".join(prompt_parts)


def _generate_sequence_prompts(
    preset_name: str,
    num_keyframes: int = 4,
) -> List[dict]:
    """
    Generate a sequence of prompts along a preset trajectory.

    Samples evenly-spaced keyframes from the trajectory, extracts
    visual vocabulary at each keyframe, and returns prompt data
    suitable for animation or storyboard workflows.

    Args:
        preset_name: Name of Phase 2.6 preset.
        num_keyframes: How many evenly-spaced frames to extract.

    Returns:
        List of keyframe dicts with step, state, vocabulary, and prompt.
    """
    trajectory = _generate_preset_trajectory(preset_name)
    total = len(trajectory)

    indices = [int(i * (total - 1) / max(1, num_keyframes - 1)) for i in range(num_keyframes)]

    keyframes = []
    for idx in indices:
        state = trajectory[idx]
        vocab = _extract_visual_vocabulary(state, strength=1.0)
        prompt = ", ".join(vocab["keywords"])
        keyframes.append(
            {
                "step": idx,
                "total_steps": total,
                "phase": round(idx / max(1, total - 1), 3),
                "state": {p: round(state[p], 4) for p in PARAMETER_NAMES},
                "visual_type": vocab["nearest_type"],
                "distance_to_type": vocab["distance"],
                "prompt": prompt,
            }
        )

    return keyframes


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

mcp = FastMCP("deformation-intent")


# ============================================================================
# LAYER 1 TOOLS - PURE TAXONOMY LOOKUP (ZERO LLM COST)
# ============================================================================


@mcp.tool()
def list_deformation_intents() -> dict:
    """
    List all available deformation intent types with descriptions.

    LAYER 1: Pure taxonomy enumeration (zero LLM cost).

    Returns overview of all deformation types that can be classified,
    along with their recommended catastrophe mappings.
    """
    return {
        "deformation_intents": {
            name: {
                "description": data["description"],
                "keywords": data["keywords"][:5],
                "recommended_catastrophe": data["recommended_catastrophe"],
            }
            for name, data in DEFORMATION_INTENTS.items()
        },
        "total_intents": len(DEFORMATION_INTENTS),
    }


@mcp.tool()
def get_intent_vocabulary(intent_type: str) -> dict:
    """
    Get complete aesthetic vocabulary for a specific deformation intent.

    LAYER 1: Pure taxonomy retrieval (zero LLM cost).

    Args:
        intent_type: One of: structural_collapse, fluid_deformation,
                    crystalline_formation, organic_growth, phase_transition,
                    mechanical_deformation, turbulent_chaos

    Returns:
        Complete vocabulary including aesthetic terms, visual vocabulary,
        and catastrophe mapping rationale.
    """
    if intent_type not in DEFORMATION_INTENTS:
        return {
            "error": f"Unknown intent type: {intent_type}",
            "available_types": list(DEFORMATION_INTENTS.keys()),
        }

    return {"intent_type": intent_type, **DEFORMATION_INTENTS[intent_type]}


@mcp.tool()
def list_intensity_profiles() -> dict:
    """
    List available intensity profiles for cascade influence.

    LAYER 1: Pure taxonomy enumeration (zero LLM cost).
    """
    return {
        "intensity_profiles": INTENSITY_PROFILES,
        "available_levels": list(INTENSITY_PROFILES.keys()),
    }


@mcp.tool()
def get_cascade_configuration() -> dict:
    """
    Get the cascade integration configuration for this brick.

    LAYER 1: Pure configuration retrieval (zero LLM cost).

    Shows how this brick connects to downstream bricks in the cascade.
    """
    return {
        "cascade_integration": CASCADE_INTEGRATION,
        "position": CASCADE_INTEGRATION["position"],
        "downstream_bricks": list(CASCADE_INTEGRATION["downstream_bricks"].keys()),
    }


# ============================================================================
# LAYER 2 TOOLS - DETERMINISTIC CLASSIFICATION (ZERO LLM COST)
# ============================================================================


@mcp.tool()
def classify_deformation_intent(prompt: str) -> dict:
    """
    Classify the deformation intent from a user prompt.

    LAYER 2: Deterministic keyword matching (zero LLM cost).

    Analyzes the prompt text and matches it against known deformation patterns
    to determine the most likely deformation type.

    Args:
        prompt: User's text description of desired transformation

    Returns:
        Classification results with intent type, confidence, and matched keywords.
    """
    result = _classify_intent(prompt)

    intent_type = result["classified_intent"]
    intent_data = DEFORMATION_INTENTS[intent_type]

    return {
        "prompt": prompt,
        **result,
        "intent_description": intent_data["description"],
        "recommended_catastrophe": intent_data["recommended_catastrophe"],
        "catastrophe_rationale": intent_data["catastrophe_rationale"],
    }


@mcp.tool()
def map_intent_to_catastrophe(intent_type: str) -> dict:
    """
    Map a deformation intent to its recommended catastrophe type.

    LAYER 2: Deterministic mapping (zero LLM cost).

    Args:
        intent_type: The classified deformation intent

    Returns:
        Catastrophe mapping with rationale and parameters.
    """
    if intent_type not in DEFORMATION_INTENTS:
        intent_type = "structural_collapse"

    intent_data = DEFORMATION_INTENTS[intent_type]

    return {
        "intent_type": intent_type,
        "recommended_catastrophe": intent_data["recommended_catastrophe"],
        "catastrophe_rationale": intent_data["catastrophe_rationale"],
        "intensity_mapping": intent_data["intensity_mapping"],
        "catastrophe_morph_params": {
            "catastrophe_type": intent_data["recommended_catastrophe"],
            "emphasis": "surface",
            "intensity_options": list(intent_data["intensity_mapping"].keys()),
        },
    }


@mcp.tool()
def recommend_brick_sequence(prompt: str) -> dict:
    """
    Recommend a sequence of processing bricks for the cascade.

    LAYER 2: Deterministic cascade routing (zero LLM cost).

    Based on the classified deformation intent, recommends which MCP servers
    (bricks) should be applied in sequence for optimal transformation.

    Args:
        prompt: User prompt to analyze

    Returns:
        Ordered list of recommended bricks with parameters.
    """
    classification = _classify_intent(prompt)
    intent_type = classification["classified_intent"]
    confidence = classification["confidence"]

    sequence = _get_cascade_sequence(intent_type, confidence)

    return {
        "prompt": prompt,
        "classification": classification,
        "recommended_sequence": sequence,
        "total_estimated_cost": "~250-450 tokens (Layer 3 synthesis only)",
        "cascade_philosophy": (
            "Layer 1-2 operations are deterministic and cost nothing. "
            "Only the final Claude synthesis steps consume tokens."
        ),
    }


# ============================================================================
# PHASE 2.6 TOOLS - RHYTHMIC DYNAMICS (ZERO LLM COST)
# ============================================================================


@mcp.tool()
def list_rhythmic_presets() -> dict:
    """
    List all Phase 2.6 rhythmic presets for deformation intent.

    LAYER 2: Pure configuration retrieval (zero LLM cost).

    Each preset defines a periodic oscillation between two deformation
    intent states in 5D parameter space, creating rhythmic aesthetic
    trajectories suitable for temporal composition.

    Returns:
        All preset configurations with periods, patterns, and descriptions.
    """
    presets = {}
    for name, config in PHASE26_PRESETS.items():
        presets[name] = {
            "state_a": config["state_a"],
            "state_b": config["state_b"],
            "pattern": config["pattern"],
            "period": config["steps_per_cycle"],
            "num_cycles": config["num_cycles"],
            "total_steps": config["num_cycles"] * config["steps_per_cycle"],
            "description": config["description"],
        }

    return {
        "presets": presets,
        "total_presets": len(presets),
        "parameter_space": {
            "dimensions": len(PARAMETER_NAMES),
            "parameters": PARAMETER_NAMES,
        },
        "available_periods": sorted(
            set(c["steps_per_cycle"] for c in PHASE26_PRESETS.values())
        ),
    }


@mcp.tool()
def get_morphospace_coordinates() -> dict:
    """
    Get the 5D parameter-space coordinates for all canonical deformation states.

    LAYER 2: Pure coordinate retrieval (zero LLM cost).

    Returns the normalized [0, 1] coordinates that define each deformation
    intent type in the morphospace. These are the vertices between which
    rhythmic presets oscillate.

    Returns:
        All state coordinates, parameter definitions, and morphospace metadata.
    """
    return {
        "parameter_names": PARAMETER_NAMES,
        "parameter_descriptions": {
            "structural_rigidity": "0.0 = fluid/amorphous → 1.0 = rigid/crystalline",
            "deformation_rate": "0.0 = gradual/static → 1.0 = sudden/explosive",
            "geometric_order": "0.0 = chaotic/organic → 1.0 = geometric/lattice",
            "force_continuity": "0.0 = discrete/impulse → 1.0 = continuous/flowing",
            "scale_complexity": "0.0 = uniform/simple → 1.0 = fractal/multi-scale",
        },
        "canonical_states": DEFORMATION_COORDS,
        "total_states": len(DEFORMATION_COORDS),
    }


@mcp.tool()
def generate_rhythmic_trajectory(
    preset_name: str,
    num_keyframes: int = 0,
) -> dict:
    """
    Generate a complete Phase 2.6 rhythmic trajectory from a preset.

    LAYER 2: Deterministic trajectory generation (zero LLM cost).

    Creates a periodic trajectory in 5D deformation parameter space by
    oscillating between two canonical intent states using the preset's
    waveform pattern.

    Args:
        preset_name: One of the preset names from list_rhythmic_presets().
        num_keyframes: If > 0, return only this many evenly-spaced keyframes
                       instead of the full trajectory. Useful for previews.

    Returns:
        Trajectory as a list of state dicts with metadata.
    """
    if preset_name not in PHASE26_PRESETS:
        return {
            "error": f"Unknown preset: {preset_name}",
            "available_presets": list(PHASE26_PRESETS.keys()),
        }

    config = PHASE26_PRESETS[preset_name]
    trajectory = _generate_preset_trajectory(preset_name)

    # Optionally subsample to keyframes
    if num_keyframes > 0 and num_keyframes < len(trajectory):
        total = len(trajectory)
        indices = [
            int(i * (total - 1) / max(1, num_keyframes - 1))
            for i in range(num_keyframes)
        ]
        sampled = [trajectory[idx] for idx in indices]
        trajectory_out = sampled
        is_subsampled = True
    else:
        trajectory_out = trajectory
        is_subsampled = False

    return {
        "preset_name": preset_name,
        "config": {
            "state_a": config["state_a"],
            "state_b": config["state_b"],
            "pattern": config["pattern"],
            "period": config["steps_per_cycle"],
            "num_cycles": config["num_cycles"],
        },
        "trajectory": [
            {p: round(s[p], 4) for p in PARAMETER_NAMES} for s in trajectory_out
        ],
        "total_steps": len(trajectory),
        "returned_steps": len(trajectory_out),
        "is_subsampled": is_subsampled,
        "cost": "0 tokens (deterministic oscillation)",
    }


@mcp.tool()
def get_domain_registry_config() -> dict:
    """
    Export domain configuration compatible with the Lushy domain registry.

    LAYER 2: Pure configuration export (zero LLM cost).

    Returns the complete domain specification needed by domain_registry.py
    for inclusion in emergent attractor discovery (Tier 4D). Includes
    parameter names, state coordinates, preset definitions, and vocabulary.

    Use this to register deformation_intent with the composition graph.

    Returns:
        Domain registry configuration dict.
    """
    presets_export = {}
    for name, config in PHASE26_PRESETS.items():
        presets_export[name] = {
            "name": name,
            "period": config["steps_per_cycle"],
            "state_a_id": config["state_a"],
            "state_b_id": config["state_b"],
            "pattern": config["pattern"],
            "description": config["description"],
        }

    vocabulary_export = {}
    for intent_type, intent_data in DEFORMATION_INTENTS.items():
        vocabulary_export[intent_type] = {
            "aesthetic_terms": intent_data["aesthetic_terms"][:5],
            "primary_forms": intent_data["visual_vocabulary"]["primary_forms"],
            "movement": intent_data["visual_vocabulary"]["movement_suggestion"],
        }

    return {
        "domain_id": "deformation_intent",
        "display_name": "Deformation Intent",
        "description": (
            "Classifies deformation aesthetics across a spectrum from "
            "structural collapse to fluid flow, crystalline order to "
            "turbulent chaos. 7 canonical states in 5D parameter space."
        ),
        "mcp_server": "deformation-intent",
        "parameter_names": PARAMETER_NAMES,
        "state_coordinates": DEFORMATION_COORDS,
        "presets": presets_export,
        "vocabulary": vocabulary_export,
        "available_periods": sorted(
            set(c["steps_per_cycle"] for c in PHASE26_PRESETS.values())
        ),
        "phase_26_status": "complete",
        "phase_27_status": "complete",
    }


# ============================================================================
# PHASE 2.7 TOOLS - ATTRACTOR VISUALIZATION PROMPTS (ZERO LLM COST)
# ============================================================================


@mcp.tool()
def list_visual_types() -> dict:
    """
    List all visual types available for prompt generation.

    LAYER 2: Pure vocabulary enumeration (zero LLM cost).

    Visual types map regions of the 5D deformation morphospace to
    image-generation keywords. Nearest-neighbour lookup translates
    any parameter coordinate to a visual vocabulary.

    Returns:
        All visual types with coordinates, keywords, and descriptions.
    """
    types = {}
    for name, data in VISUAL_TYPES.items():
        types[name] = {
            "coords": {p: round(data["coords"][p], 2) for p in PARAMETER_NAMES},
            "keywords": data["keywords"],
            "keyword_count": len(data["keywords"]),
        }

    return {
        "visual_types": types,
        "total_types": len(types),
        "parameter_space": PARAMETER_NAMES,
    }


@mcp.tool()
def extract_visual_vocabulary(
    state: str,
    strength: float = 1.0,
) -> dict:
    """
    Extract image-generation vocabulary from a deformation parameter state.

    LAYER 2: Deterministic nearest-neighbour lookup (zero LLM cost).

    Given a position in the 5D deformation morphospace, finds the nearest
    canonical visual type and returns weighted keywords suitable for
    text-to-image prompt construction.

    Args:
        state: JSON string of parameter values, e.g.
               '{"structural_rigidity": 0.5, "deformation_rate": 0.7, ...}'
               OR a canonical state name like "structural_collapse".
        strength: Keyword weight [0.0, 1.0]. Controls how many keywords
                  are included. Higher = more keywords.

    Returns:
        Nearest visual type, distance, and selected keywords.
    """
    import json as _json

    # Accept either a state name or a JSON parameter dict
    if state in DEFORMATION_COORDS:
        state_dict = DEFORMATION_COORDS[state]
    else:
        try:
            state_dict = _json.loads(state)
        except (ValueError, TypeError):
            return {
                "error": f"Invalid state. Provide a JSON parameter dict or one of: {list(DEFORMATION_COORDS.keys())}",
            }

    # Validate all parameters present
    missing = [p for p in PARAMETER_NAMES if p not in state_dict]
    if missing:
        return {"error": f"Missing parameters: {missing}", "required": PARAMETER_NAMES}

    return _extract_visual_vocabulary(state_dict, strength)


@mcp.tool()
def generate_attractor_visualization_prompt(
    state: str,
    mode: str = "composite",
    preset_name: str = "",
    num_keyframes: int = 4,
    strength: float = 1.0,
) -> dict:
    """
    Generate image-generation prompts from deformation attractor states.

    PHASE 2.7: Deterministic prompt generation (zero LLM cost).

    Translates a position (or trajectory) in the deformation morphospace
    into visual prompts suitable for text-to-image models. Three modes:

    - composite:  Single blended prompt from one state.
    - split_view: Separate prompt for the deformation domain (ready to
                  combine with prompts from other domain servers).
    - sequence:   Multiple keyframe prompts sampled along a preset
                  trajectory, suitable for animation or storyboards.

    Args:
        state: JSON parameter dict or canonical state name.
               Ignored when mode='sequence'.
        mode: 'composite', 'split_view', or 'sequence'.
        preset_name: Required when mode='sequence'. Name of a Phase 2.6
                     rhythmic preset.
        num_keyframes: Number of keyframes for 'sequence' mode (default 4).
        strength: Keyword weight [0.0, 1.0] for composite/split_view.

    Returns:
        Generated prompt(s) with metadata.
    """
    import json as _json

    if mode == "sequence":
        if not preset_name:
            return {
                "error": "preset_name is required for sequence mode",
                "available_presets": list(PHASE26_PRESETS.keys()),
            }
        if preset_name not in PHASE26_PRESETS:
            return {
                "error": f"Unknown preset: {preset_name}",
                "available_presets": list(PHASE26_PRESETS.keys()),
            }

        keyframes = _generate_sequence_prompts(preset_name, num_keyframes)
        config = PHASE26_PRESETS[preset_name]

        return {
            "mode": "sequence",
            "preset_name": preset_name,
            "description": config["description"],
            "pattern": config["pattern"],
            "period": config["steps_per_cycle"],
            "keyframes": keyframes,
            "usage_note": (
                "Each keyframe prompt captures the deformation aesthetic at "
                "that phase of the oscillation. Use as animation keyframes "
                "or as a storyboard sequence."
            ),
            "cost": "0 tokens (deterministic)",
        }

    # Resolve state for composite / split_view
    if state in DEFORMATION_COORDS:
        state_dict = DEFORMATION_COORDS[state]
    else:
        try:
            state_dict = _json.loads(state)
        except (ValueError, TypeError):
            return {
                "error": f"Invalid state. Provide JSON or one of: {list(DEFORMATION_COORDS.keys())}",
            }

    missing = [p for p in PARAMETER_NAMES if p not in state_dict]
    if missing:
        return {"error": f"Missing parameters: {missing}", "required": PARAMETER_NAMES}

    vocab = _extract_visual_vocabulary(state_dict, strength)

    if mode == "split_view":
        return {
            "mode": "split_view",
            "domain": "deformation_intent",
            "visual_type": vocab["nearest_type"],
            "distance": vocab["distance"],
            "prompt": ", ".join(vocab["keywords"]),
            "keywords": vocab["keywords"],
            "state": {p: round(state_dict[p], 4) for p in PARAMETER_NAMES},
            "note": (
                "This is the deformation_intent domain's contribution. "
                "Combine with split_view prompts from other domain servers "
                "for a multi-domain composite."
            ),
            "cost": "0 tokens (deterministic)",
        }

    # Default: composite (single-domain)
    prompt = _generate_composite_prompt(
        {"deformation_intent": state_dict},
        {"deformation_intent": strength},
    )

    return {
        "mode": "composite",
        "visual_type": vocab["nearest_type"],
        "distance": vocab["distance"],
        "prompt": prompt,
        "keywords": vocab["keywords"],
        "state": {p: round(state_dict[p], 4) for p in PARAMETER_NAMES},
        "cost": "0 tokens (deterministic)",
    }


# ============================================================================
# LAYER 3 TOOLS - SYNTHESIS PREPARATION
# ============================================================================


@mcp.tool()
def enhance_with_deformation_aesthetic(
    base_prompt: str,
    intent_type: str = "auto",
    intensity: str = "moderate",
) -> dict:
    """
    Prepare complete enhancement data for Claude synthesis.

    LAYER 3 INTERFACE: Combines Layer 1 & 2 outputs into structured
    data ready for Claude to synthesize into an enhanced prompt.

    This tool does NOT synthesize the final prompt - it provides
    all the deterministic parameters for Claude to do the creative
    synthesis, following the Lushy three-layer pattern.

    Args:
        base_prompt: Original prompt to enhance
        intent_type: Specific intent or "auto" to classify from prompt
        intensity: subtle, moderate, or dramatic

    Returns:
        Complete enhancement package for Claude synthesis.
    """
    if intent_type == "auto":
        classification = _classify_intent(base_prompt)
        intent_type = classification["classified_intent"]
        detection_info = classification
    else:
        if intent_type not in DEFORMATION_INTENTS:
            intent_type = "structural_collapse"
        detection_info = {"explicitly_selected": intent_type}

    intent_data = DEFORMATION_INTENTS[intent_type]

    if intensity not in INTENSITY_PROFILES:
        intensity = "moderate"

    intensity_profile = INTENSITY_PROFILES[intensity]

    num_terms = 2 if intensity == "subtle" else 4 if intensity == "moderate" else 6
    selected_terms = intent_data["aesthetic_terms"][:num_terms]

    intensity_description = intent_data["intensity_mapping"].get(
        "low" if intensity == "subtle" else "medium" if intensity == "moderate" else "high",
        intent_data["intensity_mapping"]["medium"],
    )

    # Phase 2.7: also include visual vocabulary from morphospace coords
    coords = DEFORMATION_COORDS.get(intent_type)
    visual_vocab = _extract_visual_vocabulary(coords) if coords else None

    return {
        "base_prompt": base_prompt,
        "detection": detection_info,
        "intent": {
            "type": intent_type,
            "description": intent_data["description"],
            "aesthetic_terms": selected_terms,
            "intensity_description": intensity_description,
        },
        "visual_vocabulary": intent_data["visual_vocabulary"],
        "morphospace_vocabulary": visual_vocab,
        "catastrophe_mapping": {
            "type": intent_data["recommended_catastrophe"],
            "rationale": intent_data["catastrophe_rationale"],
        },
        "cascade_recommendation": _get_cascade_sequence(
            intent_type, detection_info.get("confidence", 0.5)
        ),
        "synthesis_guidance": {
            "approach": (
                "Synthesize an enhanced prompt that weaves the deformation "
                "aesthetic naturally into the base prompt. Use the visual "
                "vocabulary and intensity description to guide the transformation."
            ),
            "vocabulary_to_integrate": selected_terms,
            "visual_forms": intent_data["visual_vocabulary"]["primary_forms"],
            "movement_quality": intent_data["visual_vocabulary"]["movement_suggestion"],
            "avoid": [
                "Listing deformation terms mechanically",
                "Overriding the base prompt's subject matter",
                "Using technical jargon literally",
                "Creating incoherent visual descriptions",
            ],
        },
        "cost_note": "Layer 1-2 operations: 0 tokens. Claude synthesis: ~150-250 tokens.",
    }


@mcp.tool()
def generate_cascade_prompt(
    prompt: str,
    intent_type: str = "auto",
    include_cascade_params: bool = True,
) -> dict:
    """
    Generate an enhanced prompt with cascade parameters for downstream bricks.

    LAYER 3 INTERFACE: Prepares prompt enhancement data along with
    parameters for downstream bricks in the cascade.

    Args:
        prompt: Base prompt text
        intent_type: Deformation type ("auto" to auto-detect)
        include_cascade_params: Include parameters for downstream bricks

    Returns:
        Enhanced prompt data with cascade parameters.
    """
    if intent_type == "auto":
        classification = _classify_intent(prompt)
        intent_type = classification["classified_intent"]
    else:
        if intent_type not in DEFORMATION_INTENTS:
            intent_type = "structural_collapse"
        classification = {"explicitly_selected": intent_type, "confidence": 0.8}

    intent_data = DEFORMATION_INTENTS[intent_type]

    num_terms = min(3, len(intent_data["aesthetic_terms"]))
    selected_terms = random.sample(intent_data["aesthetic_terms"], num_terms)

    result = {
        "original_prompt": prompt,
        "classified_intent": intent_type,
        "selected_aesthetic_terms": selected_terms,
        "enhanced_prompt_suggestion": f"{prompt}, {', '.join(selected_terms)}",
        "synthesis_note": (
            "The enhanced_prompt_suggestion is a simple concatenation. "
            "Claude should synthesize a more natural integration of these terms."
        ),
    }

    if include_cascade_params:
        confidence = classification.get("confidence", 0.5)
        result["cascade_params"] = {
            "catastrophe_morph": {
                "catastrophe_type": intent_data["recommended_catastrophe"],
                "intensity": "dramatic" if confidence > 0.67 else "moderate",
                "emphasis": "surface",
            },
            "diatom_morph": {
                "structure_name": None,
                "microscope_style": "SEM",
            },
        }

    return result


# ============================================================================
# SERVER INFO
# ============================================================================


@mcp.tool()
def get_server_info() -> dict:
    """
    Get comprehensive information about this MCP server.

    Returns capabilities, phase status, and integration details.
    """
    return {
        "server": "deformation-intent",
        "version": "2.7.0",
        "description": (
            "Classifies deformation intent from natural language and provides "
            "rhythmic aesthetic trajectories and visualization prompt generation "
            "for the Lushy brick cascade."
        ),
        "architecture": {
            "layer_1": "Pure taxonomy lookup from YAML olog (zero LLM cost)",
            "layer_2": "Deterministic classification, dynamics, and prompt generation (zero LLM cost)",
            "layer_3": "Structured output for Claude synthesis (~150-250 tokens)",
        },
        "phase_2_6_enhancements": {
            "rhythmic_presets": True,
            "total_presets": len(PHASE26_PRESETS),
            "available_periods": sorted(
                set(c["steps_per_cycle"] for c in PHASE26_PRESETS.values())
            ),
            "parameter_space": {
                "dimensions": len(PARAMETER_NAMES),
                "parameters": PARAMETER_NAMES,
            },
            "canonical_states": list(DEFORMATION_COORDS.keys()),
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "visual_types": list(VISUAL_TYPES.keys()),
            "prompt_modes": ["composite", "split_view", "sequence"],
            "supported_domains": ["deformation_intent"],
        },
        "tools": {
            "layer_1": [
                "list_deformation_intents",
                "get_intent_vocabulary",
                "list_intensity_profiles",
                "get_cascade_configuration",
            ],
            "layer_2": [
                "classify_deformation_intent",
                "map_intent_to_catastrophe",
                "recommend_brick_sequence",
                "list_rhythmic_presets",
                "get_morphospace_coordinates",
                "generate_rhythmic_trajectory",
                "get_domain_registry_config",
                "list_visual_types",
                "extract_visual_vocabulary",
                "generate_attractor_visualization_prompt",
            ],
            "layer_3": [
                "enhance_with_deformation_aesthetic",
                "generate_cascade_prompt",
            ],
        },
    }


# ============================================================================
# ENTRY POINT
# ============================================================================


def create_server():
    """Entry point for FastMCP Cloud deployment."""
    return mcp


if __name__ == "__main__":
    mcp.run()
