# Visual Style Baselines

Read this file when the user requests `/tutorial-doc-style handdrawn`, `/tutorial-doc-style paper`, `/tutorial-doc-style dark-paper`, or when a tutorial needs high-quality teaching figures.

This file only supplements visual style and prompt rules. The final figure generation workflow is defined in `SKILL.md`: every tutorial document handled with this skill must use `imagegen` for final teaching figures.

The current baselines include three positive styles and three anti-patterns:

1. `handdrawn`: refined hand-drawn teaching board;
2. `paper`: professional paper-style illustration;
3. `dark-paper`: dark professional paper-style illustration;
4. Anti-pattern A: unclear text, or highlight/color/hatching spills beyond the true semantic object;
5. Anti-pattern B: content is too cramped, margins are insufficient, or text touches edges or collides;
6. Anti-pattern C: a large internal title appears inside the image and steals space from the subject.

If the user provides reference images, treat them as the highest-priority style target. If image files are available, copy them into `assets/style-baselines/`; if they only appear in chat, encode reusable visual rules in this file, but first explain the purpose and content and obtain user authorization.

## 1. `handdrawn`

Use for intuition building, conceptual comparisons, mechanism-step sketches, and tutorial situations that benefit from the feeling of a careful human explanation.

Visual DNA:

- Make it feel like careful human-authored teaching notes, not a generic UI dashboard and not a casual doodle;
- Use a clean light background, readable hand-drawn linework, and restrained semantic color;
- Object boundaries, arrows, labels, and small examples should be easy to inspect at reading size;
- Colors should map to real semantics, not decoration;
- Prefer short labels and small worked examples, not long paragraphs inside the figure;
- Reader-facing explanations should be in the target language, while stable English terms keep the form used in professional contexts;
- Leave generous spacing between panels, arrows, labels, and objects.

Hard negatives:

- Do not overuse generic filled cards, stickers, dashboard chrome, or visual effects as the main language;
- Do not reduce `handdrawn` to "only the font looks handwritten while the layout remains mechanical";
- Highlights, hatching, color fills, arrows, and labels must not escape their semantic objects;
- Do not squeeze long sentences into narrow strips or let text touch edges;
- Do not use decorative sketchiness that harms structural readability.

Default tutorial style:

- When the user does not specify a visual style, prefer `handdrawn`;
- The figure should be professional, information-rich, and teachable, not childish or casual;
- Use a teaching-board metaphor: carefully drawn boxes, arrows, timelines, state changes, small examples, and concise bilingual labels when useful;
- Use the target language for reader-facing helper labels, and preserve stable English terms;
- Each figure should perform one teaching job. If crowded, split it into multiple `imagegen` figures instead of compressing labels.

Prompt skeleton:

```text
Refined hand-drawn technical tutorial diagram. Clean warm-white teaching-board background, careful human-authored composition, readable ink-like linework, restrained semantic colors, generous margins, clear arrows and state boundaries. Use short reader-facing labels plus stable English technical terms. Show concrete objects, relationships, state ownership, before/after transitions, or small worked examples according to the teaching goal. Professional, information-rich, accurate, easy to understand, lively but not decorative. No large standalone title inside the image, no dense paragraphs, no tiny text, no cramped labels, no UI dashboard cards, no stickers, no watermark, no highlight or fill outside its semantic object.
```

## 2. `paper`

Use for mechanism explanations, structural comparisons, algorithm flows, data layouts, experimental result explanations, or inspection-style summaries that need authority, precision, and a professional paper-figure feel. This mode can be sparse or dense; density should follow the teaching job.

Visual DNA:

- White or near-white background, thin black/gray linework, restrained color, clear panel structure;
- Shapes, labels, formulas, and data-flow/state-flow relationships should be precise enough for technical inspection;
- Use color only to distinguish semantic categories, boundaries, stages, or comparisons;
- Keep the composition calm and academic: balanced panels, small legends, short notes, no visual drama;
- Dense `paper` figures may include formula summaries, shape tables, legends, bilingual notes, or recap strips, but only when they support slow inspection;
- Every text block inside the figure must have a purpose; move broad explanation to the prose.

Hard negatives:

- No saturated poster colors, marketing badges, decorative icons, glossy cards, or handwritten styling;
- Do not mix multiple unrelated mechanisms into one figure;
- Do not use dense recap strips as a substitute for prose explanation;
- Do not make the first figure for a new concept so dense that it overwhelms the reader.

Prompt skeleton:

```text
Professional paper-style technical tutorial figure. White or near-white background, thin black and gray linework, restrained semantic colors, precise panel layout, clear labels, shape/table snippets only when useful, concise bilingual notes, accurate arrows and boundaries, calm academic composition. Use visual density appropriate to the teaching goal. No large standalone title inside the image, no saturated poster colors, no decorative icons, no glossy cards, no handwritten style, no cramped labels.
```

## 3. `dark-paper`

Use when a system figure, mechanism figure, communication/topology figure, deployment figure, performance-cost figure, or integrated comparison needs a dark visual atmosphere while preserving paper-style rigor. This is not cyberpunk and not dashboard style.

Visual DNA:

- Dark charcoal or near-black background, with linework and text remaining restrained and clear like a paper figure;
- Structure may use panels, pipeline, topology, timeline, or comparison layout according to the teaching purpose;
- Use a small set of semantic colors to distinguish paths, stages, boundaries, bottlenecks, risks, or comparison groups;
- Keep color semantics consistent, e.g. blue for stable path or baseline, green for local compute or successful state, orange for coordination boundary or bottleneck, red only for risk;
- Compact legends, cost cards, topology strips, or recap strips may appear, but must not compress away prose explanation;
- English technical terms can be used more freely, with short helper labels in the target language for key teaching points;
- On dark backgrounds, text must remain clear, hierarchy explicit, and whitespace generous.

Hard negatives:

- No cyberpunk decoration;
- No neon overload;
- No random code texture;
- No tiny labels;
- No dense dashboard chrome;
- Do not let dark atmosphere overpower the technical structure.

Prompt skeleton:

```text
Dark paper-style technical tutorial figure. Charcoal or near-black background, precise academic linework, restrained semantic colors, clean panel or topology composition, readable labels, concise bilingual notes, accurate arrows and boundaries. Use blue for baseline or stable paths, green for local compute or successful state, orange for coordination boundary or bottleneck, red only for risk. Professional, rigorous, calm, high-contrast, generous spacing. No large standalone title inside the image, no cyberpunk decoration, no neon overload, no random code texture, no dashboard chrome, no tiny labels.
```

## 4. Anti-Pattern A: Unclear Text Or Semantic Highlight Spillover

Problem:

- Text is blurry, too small, touching edges, or hard to read;
- Highlighting, fills, hatching, glow, or emphasis regions extend beyond the object they are supposed to represent;
- The reader cannot tell whether the colored region is the real semantic area or decoration.

Fix:

- Shorten labels, enlarge the canvas, split the figure, or move long explanations into the prose;
- Keep hatching and fills strictly inside the object, region, cell, or boundary;
- Explicitly require readable labels in the `imagegen` prompt, and forbid fill/hatching outside semantic boundaries;
- Prefer sparse internal hatching over large translucent overlays;
- After generation, ask: "Can the reader read the labels and identify the exact object, region, or state covered by the emphasis without guessing?"

## 5. Anti-Pattern B: Cramped Layout

Problem:

- Text touches edges;
- Labels collide with arrows, matrices, or panels;
- Bottom strips are too thin for their content;
- Card spacing is too narrow, making the whole figure feel tense and unprofessional.

Fix:

- Increase canvas size, split the figure, or shorten labels;
- Preserve at least 24-32 px internal padding in 1600x900 figures;
- Make bottom strips tall enough for their content, or remove them;
- Move long explanations into prose;
- Inspect at final display size and reject any text that touches edges or collides;
- After generation, ask: "Does this feel like a careful teaching note, or like content forced into a template?"

## 6. Anti-Pattern C: Large Internal Image Title

Problem:

- The top or interior of the image contains a large title line that dominates visual attention;
- The title acts like a document or section heading but adds no teaching information inside the figure;
- The title compresses the main diagram space, making readers see the title before the mechanism, structure, flow, or comparison.

Fix:

- Put titles in document headings, captions, or the prose before/after the image, not inside the final teaching figure;
- Inside the image, keep only necessary short labels, section names, object names, arrow labels, and local notes;
- Explicitly forbid large standalone title / headline inside the image in the generation prompt;
- When the figure topic needs explanation, use a prose caption such as "Figure X: Internal GPU parallel execution structure" instead of drawing that sentence across the top of the image;
- After generation, ask: "If the large title is removed, does the main content still express the teaching purpose completely?"

## 7. Mode Selection

- `handdrawn`: prefer for first intuition, conceptual contrast, and step-by-step mechanisms;
- `paper`: prefer for exact mechanisms, paper-style summaries, and inspection-heavy figures;
- `dark-paper`: prefer for dark paper-style mechanism figures, system figures, topology, deployment trade-offs, and performance-cost figures;
- Do not force one mode onto every figure, but keep style consistent within the same section or figure family.
