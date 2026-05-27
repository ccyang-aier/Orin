# Tutorial Writing Playbook

This file holds detailed writing rules that should be loaded only when needed. The main workflow and hard gates live in `SKILL.md`; this file helps the agent design stronger article structure, reader experience, source systems, and learning assessments.

## 1. Reader Experience Goals

A strong tutorial should feel like a durable learning artifact, not a chat transcript, execution log, encyclopedia entry, or figure-caption collage.

Target experience:

- The opening helps readers enter the learning context, while the specific entry point depends on the topic;
- Readers quickly understand why the topic matters;
- The first concept hook is concrete, not an abstract definition;
- Definitions compress understanding; do not treat definitions as the fixed opening for every tutorial;
- Section transitions feel natural, and the article reads like one continuous teaching process;
- Figures support the prose reasoning, and the prose should not collapse into a chain of figure explanations;
- Teaching illustrations are first-class assets: a logically clear, vivid, visually comfortable figure can significantly lower comprehension cost and make readers more willing to continue;
- Mechanism explanations should combine figures, formulas, examples, and source-backed claims;
- Central mechanisms deserve enough space; do not compress required reasoning, cost models, constraints, or failure modes into shallow asides;
- Engineering judgment should be explicit: when to use it, when not to use it, what can go wrong, and how to check it;
- References should be visible and traceable for future review.

## 2. Teaching Structure Toolkit

Treat these as teaching moves, not a fixed outline. Select, merge, reorder, or omit them according to the topic's learning problem, reader background, and publishing environment.

Before writing, rewriting, or expanding, build a teaching brief and continue only after user confirmation:

- Target reader and prior knowledge;
- Content boundary, depth standard, and explicit excluded readers, scenarios, or subdomains;
- The central question the article must answer;
- The article's distinctive teaching promise inside the broader learning path;
- Visually explainable knowledge points, visual style, figure density, and figure plan;
- Content outline;
- Learning assessment focus;
- Multi-Agent Quality Gate execution plan and required authorization.

Common structure moves follow. They are optional teaching moves, not a default chapter sequence; each tutorial should be reorganized from the confirmed brief.

1. Metadata:
   - Add frontmatter, tags, updated dates, descriptions, or similar fields only when the target environment expects them;
   - Do not force editor-specific, website-specific, or repo-specific metadata into a generic document.
2. Title:
   - Keep the user's requested title unless it is clearly temporary;
   - Use a single H1.
3. Lead-in:
   - State the core learning promise in plain language;
   - Use callouts only when the target platform supports them and they improve the reading experience.
4. Context and problem:
   - Give enough narrative runway into the problem;
   - Avoid opening by jumping into a narrow mechanism or dictionary definition;
   - Explain why this concept is forced by a system, engineering practice, or theoretical problem.
5. Definition:
   - Give the compact definition;
   - When readers have enough conceptual hooks, contrast adjacent concepts to reduce misunderstanding.
6. Mechanism:
   - Explain the core mechanism step by step;
   - Use formulas only when they clarify the mechanism;
   - Pair formulas with prose explanation and small examples.
7. Deep dive:
   - Give full sections to structurally important dependencies, costs, coordination mechanisms, constraints, or failure modes;
   - Discuss costs, boundaries, variants, and operational consequences when they affect real decisions.
8. Engineering practice:
   - Show how the concept appears in mainstream tools, commands, code paths, or deployment patterns;
   - Make constraints and trade-offs explicit.
9. Misconceptions:
   - Name common wrong mental models;
   - Correct them with crisp distinctions.
10. Summary:
   - Rebuild the whole concept in one or two paragraphs;
   - Prefer a reusable mental model over a slogan.
11. References:
   - Use one ordered list by default unless the user asks for categories.
12. Learning Assessment:
   - Substantial tutorials must add a fixed chapter named `Learning Assessment` after references;
   - Include at least 10 customized questions, mostly single-choice or multiple-choice;
   - List all questions first, then answers and explanations;
   - Cover core concepts, transfer scenarios, common pitfalls, difficult mechanisms, and implementation details.

Keep top-level teaching chapters to 6-7 or fewer, excluding references and `Learning Assessment`. Too many major chapters fragment the tutorial and weaken the article map.

Keep headings short, formal, and durable. A heading is a navigation anchor, not a compressed paragraph; avoid sentence-style headings, chatty headings, and headings that depend on the current figure order. Move explanations into the body.

## 3. Series Consistency Without Formula

Tutorial series need continuity, but upfront context should not crowd out the current task. First locate the series structure lightly, then read the minimum context needed.

Before writing a new article in a series:

- First inspect the directory structure, index, filename sequence, or navigation document to understand the series map and the current article's position;
- Do not read the full series by default; avoid exhausting context before the task itself begins;
- Choose the narrative engine by topic, such as operational story, conceptual contrast, mechanism breakdown, failure-mode investigation, decision tree, historical evolution, system lifecycle, source-code path, engineering case, or mental-model ladder.

Good pattern:

- Preserve title style, notation, figure quality, and assessment rigor, while letting each concept determine its own structure and rejecting one-size-fits-all narrative patterns.

## 4. Text-Figure Coordination

For detailed visual modes, see `visual-style-baselines.md`. This section focuses on how figures serve teaching.

Treat `imagegen` teaching figures as first-class assets, not decoration added after the prose is done. Whenever a concept, flow, structure, comparison, state change, boundary condition, or decision path can be explained or supported visually, proactively plan a figure. More figures are not the goal; the goal is for each figure to make readers more willing to read, more likely to understand, and more likely to remember.

Good figure candidates:

- Distributed architecture;
- Tensor/data flow;
- Before/after comparison;
- Memory layout;
- Communication timeline;
- System topology;
- Conceptual map;
- Lifecycle diagram.

Text-figure rules:

- Every figure needs nearby reading guidance, but the form can vary: short paragraph, caption, compact note, or surrounding explanation;
- Avoid forcing every figure into the same "Figure X reading order" template;
- Prose drives the learning path and figures support it, not the other way around;
- In long technical tutorials, three figures is usually too sparse unless the article itself is short;
- Every figure must have one clear teaching purpose;
- Figures explaining computation, data flow, state flow, or coordination must show enough structure to make the mechanism checkable;
- In English tutorials, reader-facing labels should be in English; stable professional terms should match the form readers encounter in papers, code, APIs, model cards, or framework logs;
- Avoid all-Chinese figures in English tutorials unless the figure is a screenshot or external source image;
- When a figure teaches implementation mechanics, make before/after state, ownership, data layout, communication boundaries, and constraints explicit;
- Collective operations or distributed coordination figures should show the resulting state, not merely name the operation;
- Text inside generated figures is technical content, not decoration;
- Aim for paper-level clarity: clean layout, readable labels, accurate arrows, restrained color, and no ornamental noise.

## 5. Source Standards

Build evidence before writing technical tutorials.

Prefer a mix of:

- Original or canonical papers;
- Official framework documentation;
- GitHub repositories, source docs, or design docs;
- High-quality community tutorials or long-form explanations.

Do not blur source boundaries:

- "The paper proposes/proves" is not the same as "the framework currently supports";
- "Official docs recommend" is not the same as "the community often does";
- "Local interpretation" should be written as explanation, not disguised as cited fact.

For facts that can drift, prefer current official sources and record stable links.

## 6. Tone And Depth

Use:

- Short conceptual paragraphs;
- Contextual setup before new problems;
- Concrete examples to lower abstraction cost;
- Small toy examples when formulas introduce a new algebraic distinction;
- Comparison tables only when they sharpen distinctions;
- Precise terms with clear explanation;
- "Why this design exists" framing;
- Deeper treatment for core knowledge points such as cost models, variants, constraints, and operational checks.

Avoid:

- Hype language;
- Vague adjectives without mechanism;
- Excessive quotation marks for emphasis;
- Casual or one-off headings;
- Repetitive caption templates;
- Meaningless blank lines around LaTeX;
- Overlong historical detours;
- Excessive H4/H5 nesting;
- Decorative horizontal rules;
- Unsourced and unqualified uncertain claims;
- Writing about the conversation or the agent's process.

## 7. Learning Assessment Standards

Learning assessment is part of tutorial quality, not filler.

- Questions should test real understanding, not memory alone;
- Wrong options should come from plausible misconceptions, adjacent strategies, or subtle traps;
- Do not use absurd, unrelated, or jokey distractors;
- Scenario-question options should be concrete and technically judgeable;
- Mix basic checks, transfer scenarios, and deeper reasoning;
- When multiple options seem plausible, explanations should be more detailed;
- Long assessments should provide a review path;
- After questions are drafted, run the assessment reviewer.

## 8. Durable Preferences

- Use numbered headings when the target document family expects them;
- Long-term notes must be self-contained outside the current conversation;
- Prefer source-backed explanation over unsupported confidence;
- Empty or outline-like documents should be expanded into complete artifacts rather than spawning side documents;
- AI-era tutorials treat high-quality figures as first-class learning assets;
- When figures introduce unfamiliar objects, intermediate states, implementation roles, or domain operations, include nearby explanation;
- Wherever an image can explain or support the explanation, plan an `imagegen` teaching figure where possible, while ensuring every figure has a clear teaching job;
- Bilingual labels should be intentional: the target language lowers reading cost, while English preserves standard terms readers will see in papers, docs, and code;
- Distributed, parallel, and multi-component system tutorials should explain where coordination happens, what state or layout exists after a boundary crossing, and what the cost or reliability implication is;
- Engineering-facing topics should connect theory to at least one realistic deployment or usage scenario;
- Reader-experience review must judge narrative continuity, setup, transitions, and whether the article reads like a complete tutorial rather than a collection of figure explanations;
- Generated figure factual labels must be reviewed as carefully as prose, especially when naming data ownership, state transitions, operators, protocols, or framework relationships;
- Handdrawn figures should feel like careful teaching notes: human-authored composition, readable labels, consistent semantic color, generous whitespace, and no generic sketch effects.

When the user gives a new correction, capture only reusable preferences and do not add one-off task details.
