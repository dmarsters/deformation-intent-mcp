# DeformationIntent MCP Server

Classify deformation intent from text prompts and recommend catastrophe theory transformations.

## Installation

```bash
pip install -e .
```

## Usage

```bash
python3 deformation_intent.py
```

Or import directly:

```python
from deformation_intent import classify_deformation_intent

result = classify_deformation_intent("coral structure undergoing stress")
# Returns: {
#   "classified_intent": "structural_collapse",
#   "aesthetic_terms": [...],
#   "recommended_catastrophe": "fold",
#   ...
# }
```

## Tools

- **classify_deformation_intent(prompt)** - Classify deformation type and suggest catastrophe
- **generate_cascade_prompt(prompt, intent_type)** - Generate enhanced prompt with aesthetic terms
- **recommend_brick_sequence(prompt)** - Recommend processing sequence for cascade

## Deformation Intents

- structural_collapse
- fluid_deformation
- crystalline_formation
- organic_growth
- phase_transition
- mechanical_deformation
- turbulent_chaos

## License

MIT
