# Tutorial Document Style Playbook

## 1. Desired Reader Experience

A strong tutorial should feel like a durable learning artifact. It should not read like a chat transcript, execution log, generic encyclopedia entry, or slideshow stitched together with captions.

The preferred experience is:

- The opening gives enough context and narrative runway before raising the central problem；
- The reader quickly understands why the topic matters；
- The first concept hook is concrete, not abstract；
- Definitions arrive after intuition, not before；
- Section transitions feel smooth and intentional rather than abrupt；
- Figures enrich the tutorial instead of making the prose feel like a sequence of image captions；
- Mechanism is explained with diagrams, formulas, examples, and source-backed claims working together；
- Core mechanisms receive enough depth. A central topic should not be compressed into a shallow paragraph；
- Engineering judgment is explicit: when to use it, when not to use it, and what can go wrong；
- Sources are visible at the end so claims can be traced later；

## 2. Default Tutorial Structure

Use this structure as a starting point, then adapt to the target repository, document system, or publishing environment.

1. Metadata:
   - Add frontmatter, tags, updated dates, descriptions, or source fields only when the target environment expects them；
   - Do not force editor-specific, website-specific, or repo-specific metadata into a generic document；

2. Title:
   - Keep the user's requested title unless it is clearly temporary；
   - Use a single H1；

3. Lead-in:
   - Use a callout only when the target platform supports it and it improves the reading experience；
   - State the core learning promise in plain language；

4. Context and problem section:
   - Give enough runway before the central problem；
   - Avoid abrupt openings like jumping directly into "why single GPU is not enough" without a concrete learning setup；
   - Explain what practical or conceptual problem forces this idea to exist；
   - Avoid starting with a dictionary definition；

5. Definition section:
   - Give the compact definition；
   - Contrast adjacent concepts immediately；

6. Mechanism section:
   - Explain the core mechanism step by step；
   - Use formulas only when they clarify the mechanism；
   - Pair formulas with prose explanation and small examples；

7. Deep-dive section:
   - Give central ideas enough space；
   - If a topic is structurally important, such as communication in TP, make it a full section rather than a brief aside；
   - Discuss costs, edge cases, variants, and operational consequences when they affect real decisions；

8. Engineering section:
   - Show how the concept appears in mainstream tools, commands, code paths, or deployment patterns；
   - Make constraints and trade-offs explicit；

9. Misconceptions section:
   - Include common wrong mental models；
   - Correct them with crisp distinctions；

10. Summary:
   - Rebuild the whole concept in one or two paragraphs；
   - Prefer a reusable mental model over a slogan；

11. References:
   - Use one ordered list of references unless the user asks for categories；

12. Learning assessment:
   - Add a fixed chapter named `学习测评` after references when the document is a substantial tutorial；
   - Include 10 or more customized questions, mostly single-choice or multiple-choice；
   - First list all questions, then list answers and explanations；
   - Cover core concepts, transfer scenarios, common pitfalls, difficult mechanisms, and deeper implementation details；

Keep top-level teaching chapters to 6-7 or fewer, excluding references and the learning assessment chapter. Too many major teaching chapters make the tutorial feel fragmented and make it harder for readers to see the article's full map.

Keep headings short, formal, and durable. A heading should be a navigation handle, not a compressed paragraph or casual aside. Avoid sentence-length headings such as "TP usually shards hidden dimension, output dimension, head dimension, or vocab dimension"; avoid casual headings such as "读懂 TP 的第一张图"; use short headings like "切分对象" and move the detail into body text.

## 3. Visual Strategy

Use the visual mode that best serves accuracy and learning.

Good visual candidates:

- distributed architecture；
- tensor/data flow；
- before/after comparison；
- memory layout；
- communication timeline；
- system topology；
- conceptual maps；
- lifecycle diagrams；

Figure rules:

- Every figure needs nearby reading guidance, but the guidance form should vary: sometimes a short paragraph, sometimes an ordered caption, sometimes a compact note；
- Avoid making every figure follow the same "图 X 读图顺序" template；
- The prose should drive the learning path; figures should support the prose, not the other way around；
- For long technical tutorials, three figures is usually too sparse unless the note is short. Add figures at major conceptual transitions, but only when the figure teaches a concrete point；
- A figure must have one clear teaching intent. If the reader cannot tell what the figure is proving or comparing, the figure should be redesigned；
- Matrix, tensor, and communication diagrams must show enough shape information to make the computation checkable, not merely decorative squares；
- Keep figure style consistent across a note；
- Use Chinese for reader-facing explanatory labels that reduce comprehension cost；
- Keep professional terms and standard phrases in English when they are the terms readers will see in docs or code, for example `Tensor Parallelism`, `all-reduce`, `hidden states`, `input activations`, `Column Parallel`, `Row Parallel`；
- Avoid all-English diagrams in Chinese tutorials unless the figure is a screenshot or an external source image；
- When a diagram teaches implementation mechanics, make output layout and communication semantics explicit. For example, distinguish `all-reduce -> replicated output` from `reduce-scatter -> sharded output`；
- Treat generated figure text as technical content, not decoration. If a visually good image encodes a wrong mechanism or overstates a relationship, regenerate or replace it；
- Prefer paper-level clarity: clean layout, readable labels, precise arrows, restrained color, and no ornamental noise；

Use generated raster diagrams via `imagegen` when the goal is visual intuition, architecture, conceptual comparison, or polished didactic imagery.

Use deterministic SVG/HTML/canvas/Python-rendered figures when exact text, matrix shapes, formulas, or dimensions must be reliable. For exact figures, clarity beats generative aesthetics.

### 3.1 Visual Mode Library

When the user invokes this skill with a visual mode, for example `/tutorial-doc-style handdrawn`, use the corresponding baseline style below. Modes are defaults, not cages: adapt them to the specific knowledge point while preserving their visual DNA.

#### `handdrawn`

Use for intuition-building diagrams, concept comparisons, and step-by-step mechanism sketches where a human teaching presence helps comprehension.

- Style: refined hand-drawn lecture note, clean ink and marker lines, visually comfortable, not childish；
- Background: near-white or very light warm paper only. Avoid yellow, kraft-paper, sepia, aged-paper, or stained notebook backgrounds；
- Palette: black ink plus restrained blue, green, orange, and occasional purple marker accents；
- Layout: two-panel comparisons, hand-drawn matrices, arrows, circled operators, and a compact mental-model strip work well；
- Text: Chinese-first explanatory labels with English technical terms in parentheses；
- Avoid: messy scribbles, decorative doodles, overly playful icons, dense paragraphs, or low-contrast handwriting；

#### `paper`

Use for mechanisms that need authority, exactness, and a professional paper-figure feel.

- Style: academic systems-paper figure, austere and calm, white background, thin black or gray linework；
- Palette: grayscale with very pale blue/green fills for shards and one subtle orange accent for partition or communication boundaries；
- Layout: balanced panels, visible matrix shapes, exact dimension labels, minimal legend, generous whitespace；
- Text: sparse English technical terms plus short Chinese helper labels only when useful；
- Avoid: glossy cards, saturated colors, shadows, poster-like infographics, marketing-style badges, or excessive explanatory boxes；

#### `paper-detailed`

Use when a figure must carry more technical payload than `paper`, such as formulas, shape summaries, legends, and bilingual notes.

- Style: detailed academic explainer figure, still paper-like and controlled；
- Palette: mostly black/gray with pale shard fills and restrained orange partition lines；
- Layout: two large mechanism panels plus a bottom formula/legend strip is acceptable；
- Text: allow bilingual labels and formula snippets, but keep every text block purposeful and readable；
- Use this mode for复盘图、mechanism maps, or figures that readers may inspect slowly after reading the prose；
- Avoid using this as the first figure for a brand-new concept if the density would overwhelm the reader；

#### `dark-system`

Use for runtime, deployment, topology, communication hotspot, observability, and performance-cost diagrams.

- Style: dark engineering systems figure, charcoal/near-black background, clean pipeline cards, profiler-like but not noisy；
- Palette: blue for Column Parallel or non-communication stages, green for local compute, orange for Row Parallel communication hotspots, red only for risk；
- Layout: horizontal pipeline, highlighted hotspots, compact side cost card, optional topology strip；
- Text: English technical terms are acceptable, with short Chinese helper labels for key teaching points；
- Avoid: cyberpunk decoration, neon overload, random code, dense dashboard chrome, or tiny labels；

Mode selection guidance:

- Prefer `handdrawn` for first intuition and conceptual contrast；
- Prefer `paper` for exact mathematical mechanisms；
- Prefer `paper-detailed` for summary figures and inspection-heavy diagrams；
- Prefer `dark-system` for communication, profiling, topology, and deployment trade-off diagrams；

External figure policy:

- Use paper, docs, or blog figures only when they are clearly relevant and the source is linked；
- Prefer quoting or linking the original figure when direct reuse rights are unclear；
- If a paper figure is conceptually strong but visually incomplete for the tutorial, redraw or improve it with attribution rather than copying blindly；
- Screenshots can be useful for exact source context, but they must still pass the same clarity, accuracy, and licensing/source checks；

Use Mermaid or PlantUML only for:

- deterministic flow charts；
- source-controlled architecture diagrams that will be edited frequently；
- exact protocol/state-machine diagrams；
- quick local sketches where visual polish is not important；

## 4. Source Standards

For technical tutorials, gather sources before drafting.

Use a mix of:

- original or canonical papers；
- official framework documentation；
- GitHub repositories or source-linked docs；
- high-quality community explanations or long-form blogs；

Do not blur source boundaries:

- "Paper says" is not the same as "framework currently supports"；
- "Official docs recommend" is not the same as "community often does"；
- "Local interpretation" should be written as explanation, not citation-backed fact；

When facts are likely to drift, prefer current official docs and record the access context through stable links.

Framework and API capabilities drift quickly. Verify them against current official docs before writing, and do not mix paper mechanisms, community practice, and current framework support as if they were the same kind of claim.

## 5. Tone And Depth

The preferred tone is clear, structured, and technically confident.

Use:

- short conceptual paragraphs；
- smooth transitions and contextual setup before new problems；
- concrete examples；
- small toy examples when formulas introduce a new algebraic distinction；
- comparison tables；
- precise terms with Chinese explanation；
- "why this design exists" framing；
- deeper treatment for core knowledge points: cost model, variants, constraints, and operational checks；

Avoid:

- hype language；
- vague adjectives without mechanism；
- excessive quotation marks for emphasis；
- casual or throwaway headings；
- repetitive figure-caption templates；
- unnecessary blank-line padding around LaTeX display formulas；
- overlong historical detours；
- excessive H4/H5 nesting；
- decorative horizontal rules；
- ungrounded "事实标准" claims unless sourced and qualified；
- writing about the conversation or the agent's process；

## 6. Assessment Standards

Learning assessments should be professional, not perfunctory.

- Questions should test real understanding, not simple recall only；
- Wrong choices should be plausible misconceptions, adjacent strategies, or subtle traps；
- Do not include absurd or unrelated distractors just to make the answer obvious；
- Avoid jokey or formatting-level distractors. Even wrong options should stay in the same technical neighborhood as the question；
- For scenario questions, make options concrete and technically falsifiable. Avoid vague judgments like "more stable" when the actual point is communication frequency, memory layout, topology, or framework support；
- Mix straightforward checks with transfer scenarios and deeper reasoning；
- Give detailed explanations for complex questions, especially when multiple options sound plausible；
- Include a review path when the assessment is long, so readers know which section to revisit；
- Run a dedicated assessment reviewer after drafting questions；

## 7. Iteration Notes

Current durable preferences learned from user feedback:

- Tutorial notes should use numbered headings when the target document family expects numbered headings；
- Long-term notes need stable wording and should be self-contained outside the current conversation；
- The user values source-backed explanations over unsupported confidence；
- When a note is empty or outline-like, expand it into a complete artifact rather than creating a separate side document；
- For AI-era tutorials, high-quality diagrams should be first-class educational assets, not optional decoration；
- Figure captions are mandatory when a diagram introduces unfamiliar objects such as `shard`, `hidden states`, `input activations`, or communication collectives；
- Bilingual diagram labels should be deliberate: Chinese lowers reading cost, English preserves standard technical terminology；
- Top-level chapters should normally stay within 6-7 content sections, excluding references；
- `参考资料` and `学习测评` do not count toward the 6-7 teaching-chapter limit；
- The quality gate should use content-accuracy, reader-experience, visual-quality, and assessment reviewers. Content accuracy plus reader experience should exceed 190/200, with no visual or assessment blockers；
- If reviewers find formula-to-concept jumps, add a tiny worked example instead of adding more abstract prose；
- For TP and similar systems tutorials, explicitly state where communication happens, what layout the output has after each collective, and what the communication volume implies；
- At least one realistic deployment or usage scenario should connect theory to operational judgment when the topic is engineering-facing；
- Reader-experience review must explicitly judge narrative continuity, setup, transitions, and whether the tutorial feels like a coherent article instead of a sequence of figure explanations；
- Learning assessments are part of high-quality tutorials and should be written and reviewed as their own artifact；
- Generated diagrams must be reviewed for factual labels as carefully as prose, especially when they name sharding directions, communication operators, or framework relationships；

When the user gives new correction, append only durable preferences here. Do not add transient task details.
