# Tutorial Style Playbook

## 1. Desired Reader Experience

A strong tutorial should feel like a durable learning artifact. It should not read like a chat transcript, execution log, generic encyclopedia entry, or slideshow stitched together with captions.

The preferred experience:

- The opening gives enough context and narrative runway before raising the central problem;
- The reader quickly understands why the topic matters;
- The first concept hook is concrete, not abstract;
- Definitions arrive after intuition, not before;
- Section transitions feel smooth and intentional rather than abrupt;
- Figures enrich the tutorial instead of making the prose feel like a sequence of image captions;
- Mechanism is explained with diagrams, formulas, examples, and source-backed claims working together;
- Core mechanisms receive enough depth. A central topic should not be compressed into a shallow paragraph;
- Engineering judgment is explicit: when to use it, when not to use it, and what can go wrong;
- Sources are visible at the end so claims can be traced later.

## 2. Default Tutorial Structure

Use this structure as a starting point, then adapt to the target repository, document system, or publishing environment.

1. Metadata:
   - Add frontmatter, tags, updated dates, descriptions, or source fields only when the target environment expects them;
   - Do not force editor-specific, website-specific, or repo-specific metadata into a generic document.
2. Title:
   - Keep the user's requested title unless it is clearly temporary;
   - Use a single H1.
3. Lead-in:
   - State the core learning promise in plain language;
   - Use a callout only when the target platform supports it and it improves the reading experience.
4. Context and problem:
   - Give enough runway before the central problem;
   - Avoid abrupt openings like jumping directly into "why single GPU is not enough" without a concrete learning setup;
   - Explain what practical or conceptual problem forces this idea to exist;
   - Avoid starting with a dictionary definition.
5. Definition:
   - Give the compact definition;
   - Contrast adjacent concepts immediately.
6. Mechanism:
   - Explain the core mechanism step by step;
   - Use formulas only when they clarify the mechanism;
   - Pair formulas with prose explanation and small examples.
7. Deep dive:
   - Give central ideas enough space;
   - If a topic is structurally important, such as communication in TP, make it a full section rather than a brief aside;
   - Discuss costs, edge cases, variants, and operational consequences when they affect real decisions.
8. Engineering practice:
   - Show how the concept appears in mainstream tools, commands, code paths, or deployment patterns;
   - Make constraints and trade-offs explicit.
9. Misconceptions:
   - Include common wrong mental models;
   - Correct them with crisp distinctions.
10. Summary:
   - Rebuild the whole concept in one or two paragraphs;
   - Prefer a reusable mental model over a slogan.
11. References:
   - Use one ordered list of references unless the user asks for categories.
12. Learning assessment:
   - Add a fixed chapter named `学习测评` after references for substantial tutorials;
   - Include 10 or more customized questions, mostly single-choice or multiple-choice;
   - First list all questions, then list answers and explanations;
   - Cover core concepts, transfer scenarios, common pitfalls, difficult mechanisms, and deeper implementation details.

Keep top-level teaching chapters to 6-7 or fewer, excluding references and `学习测评`. Too many major teaching chapters make the tutorial fragmented and make the article map harder to grasp.

Keep headings short, formal, and durable. A heading should be a navigation handle, not a compressed paragraph or casual aside. Avoid sentence-length headings such as "TP usually shards hidden dimension, output dimension, head dimension, or vocab dimension"; avoid casual headings such as "读懂 TP 的第一张图"; use short headings like "切分对象" and move detail into body text.

## 3. Visual Strategy

Use the visual mode that best serves accuracy and learning. For exact style rules, read `visual-style-baselines.md`.

Good visual candidates:

- distributed architecture;
- tensor/data flow;
- before/after comparison;
- memory layout;
- communication timeline;
- system topology;
- conceptual maps;
- lifecycle diagrams.

Figure rules:

- Every figure needs nearby reading guidance, but vary the form: short paragraph, ordered caption, compact note, or surrounding explanation;
- Avoid making every figure follow the same "图 X 读图顺序" template;
- The prose should drive the learning path; figures should support the prose, not the other way around;
- For long technical tutorials, three figures is usually too sparse unless the note is short;
- A figure must have one clear teaching intent. If the reader cannot tell what the figure is proving or comparing, redesign it;
- Matrix, tensor, and communication diagrams must show enough shape information to make the computation checkable;
- Use Chinese for reader-facing explanatory labels that reduce comprehension cost;
- Keep professional terms and standard phrases in English when they are the terms readers will see in docs or code, for example `Tensor Parallelism`, `all-reduce`, `hidden states`, `input activations`, `Column Parallel`, `Row Parallel`;
- Avoid all-English diagrams in Chinese tutorials unless the figure is a screenshot or external source image;
- When a diagram teaches implementation mechanics, make output layout and communication semantics explicit, for example `all-reduce -> replicated output` versus `reduce-scatter -> sharded output`;
- Treat generated figure text as technical content, not decoration;
- Prefer paper-level clarity: clean layout, readable labels, precise arrows, restrained color, and no ornamental noise.

Use generated raster diagrams via `imagegen` when the goal is visual intuition, architecture, conceptual comparison, or polished didactic imagery.

Use deterministic SVG/HTML/canvas/Python-rendered figures when exact text, matrix shapes, formulas, or dimensions must be reliable. For exact figures, clarity and style fidelity both matter; do not fall back to generic filled cards.

Use Mermaid or PlantUML only for deterministic flow charts, state machines, source-controlled diagrams, or quick local sketches where visual polish is not important.

## 4. Source Standards

For technical tutorials, gather sources before drafting.

Use a mix of:

- original or canonical papers;
- official framework documentation;
- GitHub repositories or source-linked docs;
- high-quality community explanations or long-form blogs.

Do not blur source boundaries:

- "Paper says" is not the same as "framework currently supports";
- "Official docs recommend" is not the same as "community often does";
- "Local interpretation" should be written as explanation, not citation-backed fact.

When facts are likely to drift, prefer current official docs and record stable links.

## 5. Tone And Depth

Use:

- short conceptual paragraphs;
- smooth transitions and contextual setup before new problems;
- concrete examples;
- small toy examples when formulas introduce a new algebraic distinction;
- comparison tables where they sharpen contrast;
- precise terms with Chinese explanation;
- "why this design exists" framing;
- deeper treatment for core knowledge points: cost model, variants, constraints, and operational checks.

Avoid:

- hype language;
- vague adjectives without mechanism;
- excessive quotation marks for emphasis;
- casual or throwaway headings;
- repetitive figure-caption templates;
- unnecessary blank-line padding around LaTeX display formulas;
- overlong historical detours;
- excessive H4/H5 nesting;
- decorative horizontal rules;
- ungrounded claims unless sourced and qualified;
- writing about the conversation or the agent's process.

## 6. Assessment Standards

Learning assessments should be professional, not perfunctory.

- Questions should test real understanding, not simple recall only;
- Wrong choices should be plausible misconceptions, adjacent strategies, or subtle traps;
- Do not include absurd or unrelated distractors just to make the answer obvious;
- Avoid jokey or formatting-level distractors;
- For scenario questions, make options concrete and technically falsifiable;
- Mix straightforward checks with transfer scenarios and deeper reasoning;
- Give detailed explanations for complex questions, especially when multiple options sound plausible;
- Include a review path when the assessment is long;
- Run a dedicated assessment reviewer after drafting questions.

## 7. Durable User Preferences

- Tutorial notes should use numbered headings when the target document family expects numbered headings;
- Long-term notes need stable wording and should be self-contained outside the current conversation;
- Source-backed explanations are preferred over unsupported confidence;
- When a note is empty or outline-like, expand it into a complete artifact rather than creating a separate side document;
- AI-era tutorials should treat high-quality diagrams as first-class educational assets;
- Figure captions or nearby explanations are required when a diagram introduces unfamiliar objects such as `shard`, `hidden states`, `input activations`, or communication collectives;
- Bilingual diagram labels should be deliberate: Chinese lowers reading cost, English preserves standard technical terminology;
- Top-level teaching chapters should normally stay within 6-7 content sections, excluding references and `学习测评`;
- Content accuracy plus reader experience should exceed 190/200 in the quality gate, with no visual or assessment blockers;
- If reviewers find formula-to-concept jumps, add a tiny worked example instead of more abstract prose;
- For TP and similar systems tutorials, explicitly state where communication happens, what layout the output has after each collective, and what the communication volume implies;
- At least one realistic deployment or usage scenario should connect theory to operational judgment when the topic is engineering-facing;
- Reader-experience review must explicitly judge narrative continuity, setup, transitions, and whether the tutorial feels like a coherent article instead of a sequence of figure explanations;
- Learning assessments are part of high-quality tutorials and should be written and reviewed as their own artifact;
- Generated diagrams must be reviewed for factual labels as carefully as prose, especially when they name sharding directions, communication operators, or framework relationships;
- For handdrawn diagrams, match the user's reference-board style rather than generic sketch effects: clean board-like composition, exact matrix hatching, generous spacing, and no cramped strips.

When the user gives new correction, append only durable preferences here. Do not add transient task details.
