# Personalization & Product Scope

This file is the **product-scope contract** for `comfy-api-graphs`.

It tells maintainers what this package is for and what stays out of the public
surface. It is **not** a recipe book for private stacks.

Maintained by **[ U N B R A N D E D ]**.

---

## Product intent

**What this package is**

- A small, honest Python core for building **ComfyUI API-format** workflow JSON
  (`ComfyWorkflow`, `WorkflowNode`, graph validation helpers).
- Teaching material for the community: how nodes wire (`[node_id, output_index]`),
  how `/prompt` payloads look, how KSampler / latents / checkpoints / LoRAs fit together.
- Generic FLUX / SDXL-style template **patterns** that any ComfyUI user already knows
  from the UI — expressed in code for automation and tests.
- Explicit framing: **design in the ComfyUI UI → Save (API Format) is still king.**
  This library is for automation, tests, and teaching — not a replacement for the graph editor.
- Experience-informed: built for ComfyAPI automation by people who ship graphs daily.
  Not a marketing dump of any private production pipeline.

**What this package is not**

- Not a private production stack. Private workers, orchestration hooks,
  private infra, third-party integrations, and private model stacks live elsewhere.
- Not a claim that “we published the full studio pipeline.”
- Not a fake PyPI / GitHub release story. Treat `0.1.0` as an early public-core posture:
  clone the repo, `pip install -e .`, run tests. Publish when packaging is real — don’t
  document install paths that don’t exist yet.
- Not a bidirectional sync layer with private monorepos. See [SYNC_POLICY.md](SYNC_POLICY.md).

---

## Public vs internal boundary

### PUBLIC (allowed in this repo)

| Surface | Notes |
|---------|--------|
| `ComfyWorkflow` / `WorkflowNode` | Build API JSON programmatically |
| Node-reference validation, light VRAM heuristics | Teach correct wiring before `/prompt` |
| Generic FLUX / SDXL-style templates | Community-common patterns only |
| Utility patterns (upscale / inpaint / outpaint / character refs) | Core ComfyUI nodes only |
| Quarantined video stubs (LTX / Wan) | `create()` raises `NotImplementedError` — not recipes |
| Docs that use Comfy slang | nodes, latents, checkpoints, LoRAs, custom nodes, queue, `/object_info` |
| Examples that POST to `http://127.0.0.1:8188/prompt` | Local ComfyUI only |

### INTERNAL (never document as public recipes)

Keep these out of README, templates, examples, and “how we ship” docs:

- Locked production video recipes and private quality/tuning tables
- Advanced video-mode **production** recipes beyond basic community graphs
- Private orchestration paths, secrets, private infra URLs, third-party integration internals
- Institutional model filenames, private worker JSON, or “this is how we run studio”
- Any dump of institutional checkpoints / LoRAs as the package default

If you need those, they belong in your private repos or local skills —
**as private JSON and workers**, not as “personalization overrides” of this public package.

---

## How private stacks should use this (honest model)

```
┌─────────────────────────────────────────┐
│  PUBLIC: comfy-api-graphs       │
│  Minimal shared core + generic patterns │
│  Teach API JSON / validate graphs       │
└──────────────────┬──────────────────────┘
                   │  optional, one-way consume
                   │  (ideas / tiny helpers only)
                   ▼
┌─────────────────────────────────────────┐
│  PRIVATE: your production stack         │
│  Private workflow API JSON + workers    │
│  Separate templates / local skills      │
│  NEVER reverse-sync private recipes here │
└─────────────────────────────────────────┘
```

**Recommended:** treat production as independent. If local skill code wants a tiny
`ComfyWorkflow`-style builder for throwaway tests, keep that local. Do **not** treat
this repo as the source of truth for shipping private bots / orchestrators / Comfy workers.

**Do not:**

- Bidirectionally sync “shared templates” with production workers
- Pin production on unpublished PyPI versions that don’t exist
- Checklist “private bots / orchestrators / integrations still work” as a gate for
  *this* package — those systems do not depend on this public core

---

## Personalization (without leaking filenames)

When a private fork or local skill extends the core:

1. Depend on (or vendor) only the **generic** builder API.
2. Put private checkpoints, LoRAs, CLIP/UNET names, and production graphs in **private** code.
3. Prefer exporting real graphs from ComfyUI UI (API Format) for anything that must ship.
4. Never copy production video recipes or private orchestration wiring into this repo “as examples.”

No model inventory belongs in this public document. Listing private filenames here
was a leak vector; it is intentionally gone.

---

## Version posture

| Item | Posture |
|------|---------|
| Package version | `0.1.0` — early public core, not a polished marketplace release |
| Install | Local / editable: `pip install -e ".[dev,examples]"` |
| PyPI | Do not claim until a real release exists |
| SemVer | Still useful for *this* repo’s tags; not a promise of private production pins |

---

## Related docs

- [SYNC_POLICY.md](SYNC_POLICY.md) — one-way boundary; no bidirectional fiction
- [README.md](README.md) — community-facing framing (Scope block)
- [CHANGELOG.md](CHANGELOG.md) — repositioning notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — what PRs may / may not include
