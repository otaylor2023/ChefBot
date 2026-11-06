# Recipe Planning

**Independent recipe interpretation and validation for ChefBot**

This module converts natural-language recipes into structured cooking instructions that can eventually be translated into executable robot tasks.

It handles recipe ingestion, ingredient extraction, cooking operations, intermediate food states, dependencies, and completion conditions. The resulting **Recipe Intermediate Representation (Recipe IR)** is validated against the original instructions to identify missing information, inconsistencies, and unsupported interpretations.

Recipe planning currently operates independently of ChefBot's execution agent, robot controllers, and simulation environments.

## Architecture

The system separates recipe understanding from physical execution.

```text
Recipe URL / Text
        │
        ▼
  Recipe Extraction
  (GPT-6.1 Sol)
        │
        ▼
    Recipe IR
  ├── Ingredients & quantities
  ├── Cooking operations
  ├── Intermediate food states
  ├── Equipment & containers
  ├── Dependencies
  ├── Alternatives & constraints
  └── Completion conditions
        │
        ▼
  Deterministic Validation
  ├── Evidence grounding
  ├── Schema consistency
  ├── State transitions
  ├── Dependency validation
  └── Missing information
        │
        ▼
  Optional LLM Repair
        │
        ▼
  Validated Recipe IR
        │
        ▼
  Future: Recipe → ChefBot Task Compiler
        │
        ▼
  ChefBot Execution Agent
        │
        ▼
  Robot / Simulation
```

The recipe interpreter describes **what needs to happen to ingredients**, rather than **how the robot should physically perform each operation**.

For example, `cook` represents a culinary state transformation, whereas `stir` may represent one of several physical actions required to achieve it.

This distinction allows the recipe representation to remain independent of individual robot platforms and available tools.

## Current Capabilities

| Component | Implementation |
|---|---|
| Recipe ingestion | Text files and webpage URLs |
| Structured extraction | GPT-6.1 Sol, low reasoning effort |
| Recipe representation | Pydantic-based Recipe IR |
| Cooking vocabulary | Controlled verbs with operation details |
| Ingredient tracking | Quantities, optionality, intermediate states |
| Dependency modeling | Directed operation dependencies |
| Validation | Deterministic schema, evidence, and state checks |
| Error recovery | Optional LLM-based extraction repair |
| Evaluation | Batch testing, metrics, cost and latency reporting |
| Robot execution | Not yet integrated |

The system preserves evidence from the original instructions, allowing extracted steps and assumptions to be inspected.

A valid Recipe IR does **not** guarantee that the recipe is complete, physically executable, or compatible with ChefBot's available capabilities.

## Setup

The recipe planner runs inside the existing ChefBot Python environment. No additional virtual environment or `uv` setup is required.

From the ChefBot repository:

```bash
conda activate chefbot
cd recipe_planning
```

Ensure the recipe dependencies are installed:

```bash
python -m pip install "openai>=1.66" "pydantic>=2.7,<3"
```

### API configuration

Recipe extraction uses the OpenAI API independently of ChefBot's execution agent.

| Environment variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Recipe interpretation |
| `OPENROUTER_API_KEY` | Existing ChefBot execution agent |
| `RECIPE_MODEL` | Recipe extraction model |
| `RECIPE_REASONING_EFFORT` | Model reasoning effort |

The default recipe configuration is:

```dotenv
RECIPE_MODEL=gpt-6.1-sol
RECIPE_REASONING_EFFORT=low
```

If the API keys are configured in ChefBot's root `.env`:

```bash
set -a
source ../.env
set +a
```

Never commit real API keys.

## Usage

### Interpret a local recipe

```bash
python run_recipe.py \
  --input recipes/scrambled_eggs.txt \
  --model gpt-6.1-sol \
  --reasoning-effort low
```

Or call the interpreter directly:

```python
from recipe_ir import interpret

result = interpret(
    "recipes/scrambled_eggs.txt",
    model="gpt-6.1-sol",
    reasoning_effort="low",
)

print(result.status)
print(result.recipe.ingredients)
print(result.recipe.steps)
print(result.edges)
print(result.unresolved)
```

The result contains structured ingredients, instructions, operation dependencies, predicted states, and validation findings.

### Replay a previous extraction

Existing reference outputs can be processed without an API call:

```bash
python run_recipe.py \
  --input recipes/scrambled_eggs.txt \
  --replay recipes/gold/scrambled_eggs.gold.json
```

This is useful for debugging the validation pipeline independently of model variability.

### Extract recipes from websites

The repository includes a manifest of 26 recipe URLs selected to exercise different cooking scenarios.

```text
urls.txt
```

The dataset includes simple meals, parallel preparation, optional ingredients, heating dependencies, dough preparation, baking, and multicomponent recipes.

To retrieve them:

```bash
python fetch_recipes.py urls.txt --out recipes/web
```

The scraper extracts recipe information from webpages, prioritizing structured recipe metadata when available.

Downloaded content is stored locally in `recipes/web/` and is excluded from Git.

Individual URLs can also be passed directly to the interpreter:

```python
result = interpret(
    "https://www.example.com/recipe",
    model="gpt-6.1-sol",
    reasoning_effort="low",
)
```

Webpage accessibility and extraction success depend on the source website.

## Harness: any recipe, any model (branch `recipe-harness`)

The goal and its "done" criteria are in [GOAL.md](GOAL.md).

**Models.** `--model` takes `gpt-*` (OpenAI), `openrouter:<vendor/model>` (any hosted model, `OPENROUTER_API_KEY`,
cost as reported by OpenRouter), or `ollama:<name:tag>` (local, schema-constrained decoding, `OLLAMA_HOST`).
For Ollama, `--reasoning-effort off` disables thinking; leave it empty for the model default.

**Action-ready output.** Every run folder has `actions.json`:
- `instructions` is robot-independent: per step, what is acted on (and its base ingredients), equipment and the state
  it must be in, settings, destination, `done_when` and what senses it, `after`, `stage` (same stage = parallel),
  tools (mentioned or implied, `recipe_ir/tools.yaml`), rules and open questions. It also has the recipe inventory
  and stages.
- `feasibility` is one skill profile's verdict (`skills/chefbot_sim.yaml`). Swap the profile and only this changes.

**Corpus.** `discover_recipes.py --sites sites.txt --per-site 20 --out <corpus>` grows a local corpus from the
sites' sitemaps. It honors robots.txt, waits between requests, keeps JSON-LD recipes only and writes `index.jsonl`
(difficulty features plus a heuristic `robot_plausible` label).

**Model comparison.** `eval_models.py --recipes <corpus> --subset robot_plausible --n 12 --models A B C
--reference A` runs `eval_llm.py` per model and writes `matrix.md` and `matrix.json`:
- validity, grounding and coverage
- gold accuracy, where `recipes/gold/` has a reference answer
- silver agreement with the reference model (agreement only, not correctness)
- open questions, robot-supported share, cost and latency
- the hardest recipes first

**Faithfulness.** The validator also rejects heat levels, temperatures and durations that the source does not
state near the step (`unstated_param`).

**Tests.** `python -m pytest tests -q` runs offline. Fake Ollama/OpenRouter servers serve the gold extractions,
including an end-to-end `eval_models.py` run.

## Evaluation

The evaluation pipeline measures extraction validity, source grounding, recipe structure, and additional validation findings across multiple recipes.

### Quick evaluation

```bash
python eval_llm.py \
  --recipes recipes/web \
  --model gpt-6.1-sol \
  --reasoning-effort low \
  --runs 1 \
  --limit 3
```

### Repeated evaluation

```bash
python eval_llm.py \
  --recipes recipes/web \
  --model gpt-6.1-sol \
  --reasoning-effort low \
  --runs 3 \
  --max-cost 6
```

The cost option controls evaluation scheduling using estimated or recorded costs; it should not be treated as a guaranteed API spending limit.

### Preliminary Results

Initial evaluation on October 17, 2026, using GPT-6.1 Sol with low reasoning effort:

| Metric | Result |
|---|---:|
| Recipes evaluated | 3 |
| Valid on first extraction | 1/3 |
| Valid after repair | 3/3 |
| Source grounding | 100% |
| Average extracted steps | 36.3 |
| Average output tokens | 13,109 |
| Total recorded API cost | $0.48 |
| Total wall-clock time | 238.7 seconds |

Recipes evaluated:

- Blueberry muffins
- Lentil soup
- Cinnamon roll focaccia

**Interpretation:** All three recipes produced outputs satisfying the implemented validation checks after optional repair. These results do not establish complete instruction coverage, semantic correctness, or robotic executability.

The evaluation has not yet established reliable step-level or dependency-level accuracy across the full 26-recipe dataset.

A representative evaluation summary is available at:

```text
examples/blueberry_muffins_eval.json
```

## Project Structure

```text
recipe_planning/
├── recipe_ir/
│   ├── adapters.py          # Text and URL ingestion
│   ├── api.py               # Public interpretation interface
│   ├── config.py            # Model configuration
│   ├── schema.py            # Recipe IR data models
│   ├── llm_extractor.py     # Structured LLM extraction
│   ├── recipe_pipeline.py   # Extraction and validation workflow
│   ├── checks.py            # Validation checks
│   ├── engine.py            # State and dependency processing
│   ├── compare.py           # Result comparison
│   ├── report.py            # Report generation
│   ├── runio.py             # Run serialization
│   └── verbs.yaml           # Controlled cooking vocabulary
│
├── recipes/
│   ├── gold/                # Reference extraction outputs
│   └── *.txt                # Example recipes
│
├── examples/                # Representative evaluation results
├── run_recipe.py            # Single-recipe execution
├── eval_llm.py              # Batch evaluation
├── fetch_recipes.py         # URL ingestion
├── urls.txt                 # External evaluation URLs
├── pricing.yaml             # Model pricing configuration
└── README.md
```

## Planned ChefBot Integration

The next development stage will translate validated recipes into environment-specific ChefBot tasks.

The proposed pipeline is:

```text
Validated Recipe IR
        │
        ▼
Recipe-to-Task Compiler
        │
        ├── Ingredient availability
        ├── Equipment requirements
        ├── Robot skill compatibility
        ├── Observable completion conditions
        └── Dependency preservation
        │
        ▼
ChefBot Task Specification
        │
        ▼
Existing Execution Agent
```

The compiler will need to reconcile several representations:

| Recipe representation | Required integration |
|---|---|
| Ingredients | Available physical objects and quantities |
| Containers | Kitchen fixtures, utensils, and equipment |
| Cooking verbs | Supported robot skills and tool sequences |
| Intermediate states | Observable simulator or robot states |
| Dependencies | Execution ordering and prerequisites |
| Completion conditions | Verifiable task outcomes |
| Optional steps | Capability-aware skipping or adaptation |
| Unknowns and alternatives | Explicit planning decisions |

### Design principles

**Preserve culinary semantics.** Recipe IR should represent the original cooking instructions rather than being constrained to the robot's current capabilities.

**Separate intent from execution.** A cooking operation such as `scramble` may require several physical actions, including heating, stirring, and observing food state.

**Validate before execution.** Unsupported operations, missing ingredients, and unavailable equipment should be identified before generating executable tasks.

**Maintain traceability.** Compiled tasks should retain references to their originating recipe instructions and validation results.

**Support incremental capabilities.** The initial compiler should translate a small supported vocabulary rather than attempt to execute arbitrary recipes.

## Limitations and Future Work

The current implementation has several known limitations:

- Extraction validity does not guarantee complete instruction coverage.
- Some culinary operations remain difficult to represent using the controlled verb vocabulary.
- Recipe dependencies represent logical cooking prerequisites, not necessarily physical execution constraints.
- Ingredient availability and equipment requirements are not grounded in a simulated kitchen.
- Completion conditions may describe visual or physical states that ChefBot cannot currently observe.
- Recipe interpretation does not yet generate executable ChefBot task specifications.

Future development will focus on improving extraction coverage, introducing capability-aware task compilation, and evaluating execution success within ChefBot's symbolic and RoboCasa environments.

The long-term objective is to connect **natural-language recipe understanding with reliable, observable, and recoverable robotic execution**, while maintaining clear boundaries between interpretation, planning, and physical control.
