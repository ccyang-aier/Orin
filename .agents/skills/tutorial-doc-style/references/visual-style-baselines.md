# Visual Style Baselines

Use this file whenever the user requests a visual mode such as `/tutorial-doc-style handdrawn`, `/tutorial-doc-style paper`, `/tutorial-doc-style paper-detailed`, or `/tutorial-doc-style dark-system`.

The current canonical reference board has four positive styles and three negative anti-patterns:

1. `handdrawn`: refined hand-drawn teaching board;
2. `paper`: austere paper-style mechanism figure;
3. `paper-detailed`: detailed paper-style inspection figure;
4. `dark-system`: dark engineering communication/topology figure;
5. Anti-pattern A: highlight or hatching spills beyond the actual semantic object;
6. Anti-pattern B: content is packed too tightly, with insufficient margins or cramped text;
7. Anti-pattern C: English technical labels render as missing-glyph boxes because the selected font does not cover Latin characters.

If the user supplies actual reference images in the conversation, treat those images as the highest-priority style target. If the image files are available on disk, copy them into `assets/style-baselines/` using the names above; if they are only visible in the chat, encode their visual rules here instead of pretending the binary files exist.

## 1. `handdrawn`

Use for intuition-building diagrams, concept comparisons, and step-by-step mechanism sketches where a human teaching presence helps comprehension.

Canonical visual DNA:

- The image should look like a careful human-drawn teaching note, not a generic UI dashboard and not a sloppy doodle;
- Background is near-white or very light warm paper. It must not look yellow, sepia, aged, stained, kraft-paper, or notebook-brown;
- Large title uses a confident hand-written black marker style;
- Major section titles can be blue/green/orange marker text with a single hand-drawn underline;
- Use black ink for neutral matrices, formulas, separators, and explanatory text;
- Use blue for rank 0 / column-parallel elements, green for rank 1 / row-parallel or local elements, orange for communication or reduction, and purple only for summary/mental-model strips;
- Prefer white interior panels with colored outlines over filled pastel cards;
- Use generous margins around the canvas, between panels, inside boxes, and around text;
- Use real matrix grids with exact row/column counts when shapes matter;
- Hatching should be hand-drawn and clipped inside each matrix cell or matrix rectangle. It must never spill outside the matrix boundary;
- Arrows should be simple, ink-like, and readable. Avoid decorative arrowheads or excessive crossing lines;
- Use Chinese for reader-facing explanation and English for stable technical terms such as `Column Parallel`, `Row Parallel`, `all-reduce`, `concat`, `hidden states`, `rank`, `shard`;
- Dense paragraphs do not belong inside the figure. Move long explanations to nearby prose.

Recommended layout patterns:

- Two-panel comparison with a hand-drawn vertical separator;
- One small worked example, for example `X (2x4), W (4x4), TP=2`;
- GPU/rank boxes with white interiors, colored outlines, and matrix diagrams inside;
- Bottom mental-model strip only when it has enough height and padding;
- A concise core-judgment box, such as "Column Parallel = concat" or "Row Parallel = all-reduce".

Hard bans:

- Do not use generic filled pastel cards as the main visual language;
- Do not let colored hatching or fill extend outside matrices, boxes, or semantic regions;
- Do not pack a long sentence into a thin bottom strip;
- Do not place text close to borders. Keep at least 24-32 px padding inside boxes at 1600x900 scale;
- Do not use one-line labels that collide with arrows, matrices, or borders;
- Do not make a "handdrawn" figure by merely switching to a handwriting font while keeping a dashboard layout.

Prompt skeleton for image exploration:

```text
Refined hand-drawn technical teaching diagram on near-white paper, careful black marker title, colored marker section headings, exact matrix grids, clean arrows, generous white space, blue/green/orange/purple semantic accents, Chinese helper labels with English technical terms, no decorative doodles, no yellow paper, no dense paragraphs, no cramped bottom strips, no hatching outside matrix boundaries.
```

Deterministic rendering checklist:

- Draw hatching inside each matrix cell or clip it to the matrix rectangle;
- Use outlines and sparse marker fills instead of full-card fills;
- Keep font sizes large enough for 1600px-wide images;
- Use fonts that cover Chinese, English technical terms, numbers, and punctuation. Reject missing-glyph boxes in labels such as `Column Parallel`, `all-reduce`, `rank`, and `shard`;
- Inspect at display size and at 50% scale;
- If a text block wraps more than two lines inside a figure, shorten it or move it to prose;
- Reserve bottom strips for one compact mental model only.

## 2. `paper`

Use for mechanisms that need authority, exactness, and a professional paper-figure feel.

Canonical visual DNA:

- White background, thin black or gray linework, no paper texture;
- Balanced panels with clear labels such as `(a)` and `(b)`;
- Matrix shapes and dimensions are precise;
- Use very pale blue/green fills for shards and one subtle orange dashed line for partition or communication boundaries;
- Use a restrained academic font style;
- Keep explanations sparse, preferably one sentence per panel plus a small legend;
- Use shadow only if extremely subtle; no glossy or card-heavy styling.

Hard bans:

- No saturated poster colors;
- No marketing badges;
- No handwritten fonts;
- No decorative icons;
- No dense prose blocks.

## 3. `paper-detailed`

Use when a figure must carry more technical payload than `paper`, such as formula summaries, shape tables, legends, and bilingual notes.

Canonical visual DNA:

- White or near-white background;
- Black/gray linework with pale blue/green shard fills and restrained orange partition lines;
- Two large mechanism panels plus a bottom formula/legend strip are acceptable;
- Text can be bilingual, but every text block must be purposeful and readable;
- Use this mode for recap diagrams, mechanism maps, and figures that readers may inspect slowly after reading the prose.

Hard bans:

- Do not use this as the first figure for a brand-new concept if density would overwhelm the reader;
- Do not add formula strips that are too low or too cramped;
- Do not mix many unrelated mechanisms in one figure.

## 4. `dark-system`

Use for runtime, deployment, topology, communication hotspots, observability, and performance-cost diagrams.

Canonical visual DNA:

- Dark charcoal or near-black background, not pure decorative neon;
- Horizontal pipeline layout with clean stage cards;
- Blue = Column Parallel or no-communication stage;
- Green = local compute;
- Orange = Row Parallel or communication hotspot;
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

## 5. Anti-Pattern A: Spillover Highlight

Problem:

- A matrix, shard, highlight, or diagonal hatching visually extends beyond the object it is supposed to represent;
- The reader can no longer tell whether the colored area is the actual tensor slice or just decoration.

Fix:

- Clip hatching and fills to the exact matrix rectangle or to individual cells;
- If using a deterministic renderer, draw hatching per cell or use a clipping mask;
- If using `imagegen`, explicitly forbid fill/hatching outside matrix boundaries and inspect the result;
- Prefer sparse internal hatching over large translucent overlays.

Review question:

- "Can the reader identify the exact rows/columns covered by this shard without guessing?"

## 6. Anti-Pattern B: Cramped Layout

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

## 7. Anti-Pattern C: Missing Glyphs

Problem:

- A CJK-only or Han-only font renders English terms, formulas, or punctuation as square boxes;
- The figure may look hand-drawn at a glance but becomes unusable as a technical teaching asset.

Fix:

- Choose a font family with complete Chinese, Latin, number, and punctuation coverage;
- Render one visual smoke-test image before batch generation when the figure contains bilingual labels;
- Inspect titles, section labels, formulas, and legends after rendering. Any tofu square is a blocker.

Review question:

- "Can every Chinese label, English technical term, formula, and punctuation mark be read without missing-glyph artifacts?"

## 8. Mode Selection

- Prefer `handdrawn` for first intuition and conceptual contrast;
- Prefer `paper` for exact mathematical mechanisms;
- Prefer `paper-detailed` for dense recap and inspection-heavy diagrams;
- Prefer `dark-system` for communication, profiling, topology, and deployment trade-off diagrams;
- Do not force one mode onto every figure if a different mode better serves the local teaching purpose, but keep style consistent within a section or figure family.
