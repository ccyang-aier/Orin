# Visual Style Baselines

Use this file whenever the user requests a visual mode such as `/tutorial-doc-style handdrawn`, `/tutorial-doc-style paper`, or `/tutorial-doc-style dark-system`. Treat legacy requests for `paper-detail` or `paper-detailed` as `paper` with a denser information payload, not as a separate style family.

The current canonical reference board has three positive styles and two negative anti-patterns:

1. `handdrawn`: refined hand-drawn teaching board;
2. `paper`: professional paper-style mechanism or inspection figure;
3. `dark-system`: dark engineering communication/topology figure;
4. Anti-pattern A: highlight or hatching spills beyond the actual semantic object;
5. Anti-pattern B: content is packed too tightly, with insufficient margins or cramped text.

If the user supplies actual reference images in the conversation, treat those images as the highest-priority style target. If the image files are available on disk, copy them into `assets/style-baselines/` using the names above; if they are only visible in the chat, encode their visual rules here instead of pretending the binary files exist.

Generation route for all modes:

- Every visual mode in this file is `imagegen`-first unless the user explicitly asks for a code-native/vector artifact;
- Generate `handdrawn`, `paper`, `dark-system`, and future style modes with `imagegen` as the primary visual source;
- Do not use Python/PIL/SVG/canvas as the primary renderer for tutorial figures governed by this skill. These tools may be used only as an explicitly disclosed correction or annotation layer after an `imagegen` draft exists;
- If technical labels or structural details need correction after generation, use prompt iteration, reference images, selective editing, or post-processing after the `imagegen` draft. Do not silently change the generation medium.

## 1. `handdrawn`

Use for intuition-building diagrams, concept comparisons, and step-by-step mechanism sketches where a human teaching presence helps comprehension.

Canonical visual DNA:

- Make the image feel like a careful human teaching note, not a generic UI dashboard and not a sloppy doodle;
- Use a clean light background, readable hand-drawn linework, and restrained semantic color;
- Keep object boundaries, arrows, labels, and examples easy to inspect at reading size;
- Use color consistently to mean real concepts in the diagram, not decoration;
- Prefer short labels and small worked examples over dense paragraphs inside the figure;
- Use Chinese for reader-facing explanations and English for stable technical terms when those terms are the professional vocabulary readers will encounter;
- Give the composition generous margins and enough room between panels, arrows, labels, and objects.

Hard bans:

- Do not use generic filled cards, decorative stickers, dashboard chrome, or visual effects as the main language;
- Do not make "handdrawn" mean only a handwriting-like label style while the layout remains mechanical;
- Do not let highlights, hatching, fills, arrows, or labels escape the semantic object they describe;
- Do not squeeze long sentences into thin strips or place text close to borders;
- Do not use decorative sketchiness that makes technical structure harder to read.

Prompt skeleton for image exploration:

```text
Refined hand-drawn technical teaching diagram, clean light background, human-authored teaching-board composition, readable ink-like lines, restrained semantic color, short Chinese helper labels with stable English technical terms, clear arrows, generous spacing, no decorative doodles, no dense paragraphs, no cramped labels, no highlight or fill outside its semantic object.
```

## 2. `paper`

Use for mechanisms that need authority, exactness, and a professional paper-figure feel. This mode covers both sparse mechanism figures and denser inspection figures; choose the density according to the teaching job instead of switching to a separate style.

Canonical visual DNA:

- Use a white or near-white background, thin black or gray linework, restrained color, and clear panel structure;
- Make shapes, labels, formulas, and data-flow/state-flow relationships precise enough for technical inspection;
- Use subtle color only to distinguish semantic categories, boundaries, stages, or comparisons;
- Keep the composition calm and academic: balanced panels, small legends, concise notes, and no visual drama;
- A denser `paper` figure may include formula summaries, shape tables, legends, bilingual notes, or recap strips when they help slow inspection;
- Keep every text block purposeful and readable. Move broad explanation to nearby prose.

Hard bans:

- No saturated poster colors, marketing badges, decorative icons, glossy cards, or handwritten styling;
- Do not mix many unrelated mechanisms in one figure;
- Do not use dense recap strips as a substitute for explanation in the tutorial body;
- Do not make the first figure for a new concept so dense that it overwhelms the reader.

## 3. `dark-system`

Use for runtime, deployment, topology, communication hotspots, observability, and performance-cost diagrams.

Canonical visual DNA:

- Dark charcoal or near-black background, not pure decorative neon;
- Horizontal pipeline layout with clean stage cards;
- Blue = stable path, baseline flow, or no-coordination stage;
- Green = local compute, local state, or successful handoff;
- Orange = coordination boundary, bottleneck, or communication hotspot;
- Red = risk only;
- Use compact cost card and optional topology strip;
- Use English technical terms freely, with short Chinese helper labels for key teaching points;
- Keep cards aligned and give each stage enough breathing room.

Hard bans:

- No cyberpunk decoration;
- No neon overload;
- No random code texture;
- No tiny labels;
- No dense dashboard chrome.

## 4. Anti-Pattern A: Spillover Highlight

Problem:

- A highlight, fill, hatching, glow, or emphasis region visually extends beyond the object it is supposed to represent;
- The reader can no longer tell whether the colored area is the actual semantic region or just decoration.

Fix:

- Clip hatching and fills to the exact object, region, cell, or boundary they represent;
- If using a deterministic post-processing renderer, draw repeated fills within each object or use a clipping mask;
- If using `imagegen`, explicitly forbid fill/hatching outside semantic boundaries and inspect the result;
- Prefer sparse internal hatching over large translucent overlays.

Review question:

- "Can the reader identify the exact object, region, or state covered by this visual emphasis without guessing?"

## 5. Anti-Pattern B: Cramped Layout

Problem:

- Text sits too close to borders;
- Labels collide with arrows or matrices;
- Bottom strips are too thin for their content;
- Inter-card spacing is too narrow, making the whole diagram feel tense and unprofessional.

Fix:

- Increase canvas space, split the figure, or shorten labels;
- Use 24-32 px minimum inner padding for 1600x900 diagrams;
- Keep bottom strips tall enough for the actual text, or remove them;
- Reserve long explanations for nearby prose;
- Inspect at the final display size and reject any text that touches borders or other elements.

Review question:

- "Does the figure feel like a careful teaching note with breathing room, or like content squeezed into a template?"

## 6. Mode Selection

- Prefer `handdrawn` for first intuition and conceptual contrast;
- Prefer `paper` for exact mechanisms, professional paper-style summaries, and inspection-heavy diagrams;
- Prefer `dark-system` for communication, profiling, topology, and deployment trade-off diagrams;
- Do not force one mode onto every figure if a different mode better serves the local teaching purpose, but keep style consistent within a section or figure family.
