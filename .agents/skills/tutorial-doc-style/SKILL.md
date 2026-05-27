---
name: tutorial-doc-style
description: "Write, revise, and iteratively improve source-backed technical tutorials, especially for AI/LLM engineering tutorials, long-term knowledge-base articles, Markdown notes, repository docs, image-rich tutorials, learning assessments, multi-agent quality gates, and user-feedback-driven style refinement."
---

# Technical Tutorial Style

Turn tutorials into durable learning artifacts, not one-off answers. Resolve the target language from the user request or target document; default to English only when neither indicates another language. Preserve stable professional English terms in any target language.

## 1. Non-Negotiable Rules

- When the user explicitly invokes this skill, or the task clearly involves high-quality tutorial writing, revision, expansion, structural rewriting, or image-rich tutorial design, follow this workflow;
- Brief confirmation is a hard execution gate: after only the minimal orientation needed to identify the target artifact, series position, and obvious repository constraints, output the teaching brief and stop; do not continue the same turn into evidence gathering, detailed figure design, `imagegen`, drafting, rewriting, expansion, target-document edits, learning-assessment work, or quality-gate execution;
- Every tutorial document handled with this skill must complete direction-setting and produce a teaching brief before drafting, rewriting, expanding, or generating teaching assets; the user must explicitly confirm or revise the brief before any downstream work continues;
- The teaching brief must include at least audience, boundaries, depth, visual style, exclusions, content outline, figure plan, and assessment focus;
- Every tutorial document handled with this skill must use `imagegen` for final teaching figures, and must complete figure planning, generation, review, local saving, and document references before finalization;
- Teaching-figure creation and correction must rely only on the `imagegen` skill; do not use scripts, drawing tools, image editors, SVG/HTML/canvas reconstruction, raster/vector cleanup, or any other post-generation processing to repair imperfect generated figures;
- `imagegen` teaching figures are first-class tutorial assets; mechanisms, flows, structures, comparisons, state changes, and decision paths that can be explained or supported visually should receive high-quality figure planning wherever possible;
- Every tutorial document handled with this skill must run the Multi-Agent Quality Gate; if tool policy requires user authorization, request it during the teaching brief stage; if tools are unavailable or authorization is denied, report the blocker explicitly and do not silently skip it;
- Do not send a final completion response while required image assets, path checks, multi-agent review, learning assessment, or Markdown validation gates remain unmet;

## 2. Workflow

1. Set direction and confirm the brief:
   - Before the brief, do only minimal orientation: read the user request, repository instructions, target path, target document header or short excerpt, and series filenames or index if needed; do not perform broad research, source tracing, detailed outline expansion, final figure prompting, image generation, or document edits;
   - Ask 2-4 high-value questions when answers are necessary to avoid a risky assumption; if the user already provided enough direction, state reasonable assumptions inside the brief instead of blocking;
   - Produce a teaching brief: target reader, exclusions, content boundary, depth standard, central question, distinctive teaching promise, content outline, visually explainable knowledge points, figure density and figure plan, assessment focus, and quality gate execution plan;
   - End the response by asking the user to confirm or revise the brief, then stop work for that turn;
   - The user must explicitly confirm the brief before tutorial writing, rewriting, expansion, evidence gathering beyond orientation, detailed figure design, `imagegen`, learning-assessment work, quality-gate execution, or target-document edits continue;
   - Treat ambiguous replies as not yet confirmed; clarify or update the brief and wait again;
   - Honor the user's exclusions and do not make unrelated subdomains, toolchains, or workflows the article spine.

2. After confirmation, load context:
   - Read repository, project, `AGENTS.md`, or equivalent instructions;
   - Open the target document and preserve the user's existing structure unless it clearly harms tutorial quality;
   - If the target document belongs to a series, first inspect the directory structure, index, or filenames; do not read the full series context by default;
   - For substantial writing, style revisions, repeat iterations, or reader-experience issues, read `references/tutorial-style-playbook.md`;
   - For figure-heavy work, tutorial figures, named visual modes, or `/tutorial-doc-style <mode>` requests, read `references/visual-style-baselines.md`.

3. Build evidence:
   - Search for sources when the tutorial depends on external technical facts;
   - Prefer a balanced source set: papers/arXiv, official documentation, GitHub/source documentation, and high-quality community tutorials;
   - Keep paper claims, official capabilities, community practice, and local interpretation separate;
   - Verify drift-prone framework/API capabilities against current official sources.

4. Refine the teaching path:
   - Use the user-confirmed teaching brief and content outline as the basis for the teaching path;
   - Do not apply a fixed chapter template, fixed opening pattern, fixed definition order, or fixed figure rhythm to every tutorial;
   - Choose the narrative form that best fits the current topic, such as conceptual contrast, mechanism breakdown, system lifecycle, failure-mode investigation, decision tree, source-code path, or engineering case;
   - Select the necessary teaching layers from intuition, mechanism, implementation details, cost model, constraints, failure modes, and decision rules, then reorder them for the current topic;
   - Treat central mechanisms as full teaching objects and do not compress them into shallow asides;
   - Avoid formulaic tutorial series structure. Chapter order, examples, figure strategy, and decision framework must serve the current concept;
   - Keep top-level teaching chapters to 6-7 or fewer, excluding references and `Learning Assessment`;
   - Keep headings short, formal, and durable.

5. Generate and review teaching figures:
   - Treat teaching figures as first-class tutorial assets; a logically clear, vivid, visually comfortable figure often motivates readers better than long prose;
   - Plan figures wherever an image can directly explain or support the explanation, while ensuring every figure has a clear teaching job;
   - Define each figure's teaching purpose, visual mode, required labels, and target path;
   - Named visual modes include `handdrawn`, `paper`, `dark-paper`, and future modes; final assets always use `imagegen`;
   - If the user provides reference images, treat them as the highest-priority style target; copy available files into `assets/style-baselines/`, and encode chat-only images as text rules;
   - If generated figures have wrong labels, cramped text, layout issues, or incorrect structure, fix them only with clearer `imagegen` prompts, reference images, shorter labels, split figures, or full regeneration; never repair a generated figure through post-processing or external editing;
   - Review every figure for teaching purpose, factual labels, structural boundaries, semantic highlighting, visual hierarchy, whitespace, style consistency, and reject large visually dominant titles inside the image;
   - Save final images into the project image/assets folder and reference them correctly from the target document.

6. Write and finalize:
   - The document must be self-contained outside the current conversation;
   - Follow the target repository or publishing environment for frontmatter, numbered headings, callouts, punctuation, and citation style;
   - Use the target document language for reader-facing prose, captions, and generated-figure helper labels; preserve stable professional English terms such as framework concepts, operators, APIs, model names, paper terms, and standard technical phrases;
   - Avoid process narration, chat traces, excessive quotation marks, repetitive caption templates, meaningless blank lines, and decorative horizontal rules.

## 3. Learning Assessment

Substantial tutorials must add a fixed final chapter named `Learning Assessment` after references.

- Include at least 10 customized questions;
- Mostly use single-choice and multiple-choice questions;
- List all questions first, then answers and explanations;
- Wrong options must be plausible misconceptions, adjacent strategies, or subtle traps;
- Do not use absurd, jokey, or unrelated distractors;
- Give richer explanations for difficult or trap questions.

## 4. Multi-Agent Quality Gate

Every tutorial document handled with this skill must run the quality gate after the main draft is ready. If the current environment requires explicit user authorization for multi-agent tools, request it during the teaching brief stage.

1. Spawn these agents in parallel:
   - Content accuracy reviewer: factual accuracy, source support, terminology, conceptual order;
   - Reader experience reviewer: clarity, continuity, transitions, depth, and whether the article is excellent enough;
   - Visual quality reviewer: figure purpose, technical correctness, style consistency, spacing, labels, and anti-patterns;
   - Assessment author: draft the `Learning Assessment` chapter.

2. Ask reviewers to return:
   - Score out of 100;
   - Top risks;
   - Concrete revision suggestions;
   - Pass/fail.

3. After integrating assessment questions, spawn an assessment reviewer to check correctness, coverage, ambiguity, and difficulty balance.

4. Passing standard:
   - Content accuracy + reader experience + visual reviewer must exceed 285/300;
   - Assessment reviewer must report no correctness blocker.

## 5. Validation

Before completion, explicitly check:

- Final project image assets exist;
- Every generated figure is referenced from the target document;
- `git diff --check -- <target-file>` passes;
- If this skill was modified, run the skill validator;
- Multi-Agent Quality Gate has passed, or a tool/authorization blocker has been explicitly reported;
- `Learning Assessment` is complete and has passed assessment review.

## 6. Feedback Capture

When the user criticizes or requests changes:

1. Classify feedback as structure, depth, tone, visual strategy, source standard, format, assessment, or examples;
2. Fix the current artifact first;
3. If the feedback is reusable, update `references/tutorial-style-playbook.md` or `references/visual-style-baselines.md`; before updating, explain the purpose and content to the user and request authorization;
4. Keep this `SKILL.md` concise and place evolving detail rules in references.

## References

- `references/tutorial-style-playbook.md`: teaching structure, reader experience, source standards, assessment standards, and durable user preferences;
- `references/visual-style-baselines.md`: visual modes, reference styles, prompt rules, and figure anti-patterns.
