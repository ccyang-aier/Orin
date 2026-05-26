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

## 2. Structure Design Toolkit

Use these as teaching moves, not as a fixed outline. Select, reorder, combine, or omit them according to the topic's learning problem, target reader, and publishing environment.

Before using any structure, establish a short teaching brief:

- target reader and their prior knowledge;
- excluded audience or domain;
- central question the article must answer;
- the article's distinctive teaching promise within the broader learning path;
- desired visual style and figure density;
- expected assessment style.

A tutorial series should feel coherent without becoming formulaic. Keep recurring terminology, notation, tone, and quality standards where they help readers, but let each topic determine its own narrative architecture, examples, figure strategy, and assessment focus.

Possible structural moves:

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
   - Avoid abrupt openings, such as jumping into a narrow problem or starting the knowledge-system exposition before the reader has a concrete learning setup;
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
   - If a topic is structurally important, such as a required dependency, cost model, coordination mechanism, or operational constraint, make it a full section rather than a brief aside;
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

Keep headings short, formal, and durable. A heading should be a navigation handle, not a compressed paragraph or casual aside. Avoid sentence-length headings that try to explain the whole mechanism; avoid casual headings that depend on the current draft's figure order or conversational framing; use short conceptual labels and move detail into body text.

## 2.1 Direction-Setting Questions

When the user has not fully specified the tutorial direction, ask a small number of high-yield questions before drafting. Do not ask a long questionnaire. Good defaults:

1. Who is this tutorial for, and who is it explicitly not for?
2. What should the article help readers decide or understand after reading?
3. Should this article emphasize intuition, implementation mechanics, engineering deployment, math, or conceptual comparison?
4. What visual style should the diagrams follow: refined hand-drawn teaching board, paper-style mechanism figure, dark system topology, or a user-provided reference?

If the user gives a clear exclusion, honor it structurally. For example, if a DP tutorial is not for training-focused readers, do not make training loops, optimizer states, backward passes, or ZeRO/FSDP the central spine. Mention them only as contrast or boundary notes when they are necessary for conceptual clarity.

## 2.2 Series Cohesion Without Formula

Before writing a new article in an existing series:

- Read the series context to learn tone, terminology, notation, and quality bar;
- Identify whether a shared chapter pattern is genuinely useful for the current topic or merely habitual;
- Write a one-sentence "distinctive promise" for the new article;
- Pick the narrative engine that best fits the topic:
  - operational story;
  - conceptual contrast;
  - failure-mode investigation;
  - decision tree;
  - historical evolution;
  - system lifecycle;
  - mental-model ladder.

Bad pattern:

- Every installment follows the same chapter sequence regardless of the concept's actual explanatory needs.

Better pattern:

- Keep series continuity in title style, notation, figure quality, and assessment rigor, while letting each topic choose its own structure.

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
- Diagrams that explain computation, data flow, state flow, or coordination must show enough structural information to make the mechanism checkable;
- Use Chinese for reader-facing explanatory labels that reduce comprehension cost;
- Keep professional terms and standard phrases in English when they are the terms readers will see in papers, docs, code, APIs, model cards, or framework logs;
- Avoid all-English diagrams in Chinese tutorials unless the figure is a screenshot or external source image;
- When a diagram teaches implementation mechanics, make before/after state, ownership, data layout, communication, and boundary conditions explicit. If collective operations or distributed coordination are involved, distinguish the resulting state rather than only naming the operation;
- Treat generated figure text as technical content, not decoration;
- Prefer paper-level clarity: clean layout, readable labels, precise arrows, restrained color, and no ornamental noise.

Use generated raster diagrams via `imagegen` when the goal is visual intuition, architecture, conceptual comparison, or polished didactic imagery.

Do not use deterministic SVG/HTML/canvas/Python/PIL-rendered figures, plotting libraries, scripts, Mermaid, PlantUML, or local annotation layers for final tutorial images governed by this skill. If an `imagegen` result has bad labels, cramped layout, or incorrect structure, regenerate it with a clearer prompt, split the figure, shorten labels, or ask a direction-setting question before continuing.

Do not include Mermaid or PlantUML diagrams as substitutes for tutorial illustrations in image-rich AI/LLM tutorial documents. They may appear only when the user explicitly requests a code-native diagram deliverable separate from this skill's final tutorial-image workflow.

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
- Figure captions or nearby explanations are required when a diagram introduces unfamiliar objects, intermediate states, implementation roles, or domain-specific operations;
- Bilingual diagram labels should be deliberate: Chinese lowers reading cost, English preserves standard technical terminology when those terms are what readers will encounter in papers, docs, or code;
- Top-level teaching chapters should normally stay within 6-7 content sections, excluding references and `学习测评`;
- Content accuracy plus reader experience should exceed 190/200 in the quality gate, with no visual or assessment blockers;
- If reviewers find formula-to-concept jumps, add a tiny worked example instead of more abstract prose;
- For distributed, parallel, or multi-component systems tutorials, explicitly state where coordination happens, what state or layout exists after each boundary crossing, and what the cost or reliability implication is;
- At least one realistic deployment or usage scenario should connect theory to operational judgment when the topic is engineering-facing;
- Reader-experience review must explicitly judge narrative continuity, setup, transitions, and whether the tutorial feels like a coherent article instead of a sequence of figure explanations;
- Learning assessments are part of high-quality tutorials and should be written and reviewed as their own artifact;
- Generated diagrams must be reviewed for factual labels as carefully as prose, especially when they name data ownership, state transitions, operators, protocols, or framework relationships;
- For handdrawn diagrams, prioritize a careful teaching-board feel: human-authored composition, legible labels, consistent semantic color, generous spacing, and no generic sketch effects.
- Do not generate final tutorial images with scripts, Python/PIL, SVG, HTML/canvas, Mermaid, PlantUML, or local post-processing. Use `imagegen` iteration as the default correction path.
- Avoid formulaic repetition across tutorial series. Preserve continuity without forcing the same article architecture onto every topic.
- Before drafting, quickly confirm audience, exclusions, narrative focus, and visual style when the user has not specified them.

When the user gives new correction, append only durable preferences here. Do not add transient task details.
