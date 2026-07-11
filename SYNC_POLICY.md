# Sync Policy

## Verdict

**There is no bidirectional sync.**

`comfy-api-graphs` is a **minimal public core**. Private production stacks
(workflow API JSON, orchestration workers, private templates) are separate. They are
related by *ideas*, not by a shared release train.

Maintained by **[ U N B R A N D E D ]**.

If older notes talked about “sync both ways,” “pin production to PyPI,” or
“verify private bots / orchestrators / integrations after bumping this package,”
treat those as **retired fiction**. This file replaces them.

---

## Two surfaces

| Surface | Owns | Source of truth |
|---------|------|-----------------|
| **Public package** (this repo) | Tiny `ComfyWorkflow` builder, graph validation, generic FLUX/SDXL-style patterns, teaching docs | This git repo alone |
| **Private production** | Private ComfyUI API JSON, workers, private model stacks | Private repos — **not** this package |

```
Public core  ──optional one-way ideas──▶  private experiments / local skills
Production private JSON  ──✗ NEVER──▶  public recipes
```

---

## What may move public → private (optional, rare)

Only **generic** improvements, and only when someone consciously ports them:

- Bug fixes in `ComfyWorkflow` / `WorkflowNode` wiring
- Safer node-reference validation
- Clearer docs on API-format JSON, `/prompt`, `[node_id, output_index]`
- Community-safe FLUX/SDXL graph patterns (no institutional filenames)

Port by hand. Test in the private tree. Do not automate. Do not treat this package
as a production dependency that must stay in lockstep.

---

## What must never move private → public

- Locked production video recipes and private quality/tuning tables
- Advanced video-mode production recipes beyond basic community graphs
- Private orchestration paths, secrets, private infra URLs, third-party integration internals
- Institutional model filenames, private worker JSON, institutional LoRAs
- Any “this is how we ship studio” dump disguised as a template

If a fix was found in production, extract the *generic* lesson (e.g. “broken
edge refs”) and re-implement it cleanly here — without the private knobs.

---

## What this policy ends

| Old claim | Replacement |
|-----------|-------------|
| Sync both ways | One optional direction: public ideas → private use |
| Shared template list with production | Public has teaching templates; production has private JSON |
| Pin private stacks on `comfy-api-graphs==x.y.z` from PyPI | Production does not run on this package’s release train |
| Staging gate: private bots / orchestrators after sync | Irrelevant to this repo’s lifecycle |
| Emergency rollback via `pip install previous` in private stacks | Rollback = restore private workers / DBs — elsewhere |

---

## Forbidden operations

1. Do not auto-sync any tree with private monorepos.
2. Do not publish private model names or production video recipes as “examples.”
3. Do not document PyPI / production pins that do not exist.
4. Do not modify this public core inside production “in place” and push back.
5. Do not claim this library executes ComfyUI jobs — it builds JSON; ComfyUI’s
   queue (`/prompt`) runs them.

---

## Versioning (this repo only)

- Semantic versioning applies to **this** package’s tags and CHANGELOG.
- `0.1.0` = early public-core posture, not a marketplace launch.
- Real PyPI publishing is a separate decision; until then, install from source:
  `pip install -e ".[dev,examples]"`.

---

## Questions

- Product scope & public/internal boundary → [PERSONALIZATION.md](PERSONALIZATION.md)
- Community install / usage framing → [README.md](README.md)
- Changelog of the repositioning → [CHANGELOG.md](CHANGELOG.md)
