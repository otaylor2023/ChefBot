# Recipe URL ingestion (standalone)

Run in the ChefBot repository root with its recipe dependencies installed and `OPENAI_API_KEY` configured for extraction (the scraper itself does not call the LLM). The existing `OPENROUTER_API_KEY` belongs to the independent ChefBot execution agent. Never commit `.env` with actual keys.

```bash
# Copy the template and replace comments with actual recipe URLs, one per line.
cp urls.example.txt urls.txt
# Open urls.txt in an editor and add URLs.
python fetch_recipes.py urls.txt --out recipes/web
# Verify actual files downloaded
find recipes/web -maxdepth 1 -name '*.txt' | wc -l

# Test the first three saved recipes (requires OPENAI_API_KEY).
python eval_llm.py --recipes recipes/web --runs 1 --limit 3
# Larger evaluation, with recorded-spend guard, not a hard budget cap
python eval_llm.py --recipes recipes/web --runs 3 --max-cost 6

# Also supported: direct extraction from one URL
python run_recipe.py --input 'https://site.example/recipe' --show-trace
```

By default extraction uses `gpt-6.1-sol` with `low` reasoning. Override using `RECIPE_MODEL`, `RECIPE_REASONING_EFFORT`, or `--model` and `--reasoning-effort`. Export `.env` to the shell explicitly if using environment-file configuration (`set -a; source .env; set +a`).

The URL scraper is the existing original `fetch_recipes.py`. It uses `recipe_ir.adapters.from_url`, requires schema.org Recipe JSON-LD, skips pages without that structured metadata, and stores clean text and URL metadata under `recipes/web/`. Some sites block automation or lack JSON-LD; those are reported as skipped. Respect each site's usage policies and avoid high-frequency requests. `urls.txt` can contain local HTML paths. To retain only URL citations without reproducing site content in Git, keep `recipes/web/` untracked.
