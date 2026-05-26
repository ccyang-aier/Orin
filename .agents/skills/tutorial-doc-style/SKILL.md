---
name: "tutorial-doc-style"
description: "Write, revise, and iteratively improve source-backed technical tutorial documents, especially Chinese AI/LLM engineering tutorials that need clear narrative structure, rigorous concepts, high-quality educational diagrams, professional learning assessments, explicit sources, and user-feedback-driven style refinement. Use for Markdown notes, knowledge-base articles, repo docs, or long-form tutorials; adapt to the target project's conventions instead of assuming any specific workspace or editor."
---

# Tutorial Document Style

Use this skill when a tutorial should become a polished long-term learning artifact rather than a one-off answer.

## Workflow

0. Apply hard gates for substantial tutorials:
   - Treat this workflow as mandatory, not advisory, when the user invokes this skill or the task clearly matches it;
   - Before drafting a substantial tutorial, run a brief direction-setting exchange with the user unless the request already specifies audience, topic boundary, desired depth, visual style, and exclusions;
   - For Chinese AI/LLM tutorial work, plan project-bound tutorial figures before finalizing the prose, and use `imagegen` as the only generation path for final figure assets;
   - Never use Python/PIL, SVG, HTML/canvas, Mermaid, PlantUML, plotting libraries, scripts, or other deterministic/local rendering to generate, redraw, reconstruct, typeset, annotate, or post-process final tutorial images. These tools may be used only for non-visual file operations such as copying, renaming, path checks, or metadata inspection;
   - For substantial revisions, run the Multi-Agent Quality Gate before completion whenever sub-agent tools are available and permitted;
   - If tool policy requires explicit user authorization for sub-agents and the user has not provided it, pause and ask for that authorization instead of silently skipping the gate;
   - Do not send a final answer while any required figure generation, image-path verification, reviewer integration, assessment review, or Markdown validation gate is unmet.

1. Set direction before drafting:
   - Ask 2-4 concise questions when the tutorial direction is under-specified. Prefer questions that determine audience, scope exclusions, narrative angle, diagram style, and the article's role in a broader learning path;
   - Explicitly confirm excluded audiences or domains. If the user says the tutorial is not for training engineers, do not center training workflows, optimizer state, backward pass, or training-only frameworks unless they are needed for a brief contrast;
   - If the document belongs to a series, preserve continuity in terminology, notation, tone, and quality bar while designing the current article's structure from its own teaching goal, conceptual tension, and reader journey;
   - Produce a short teaching brief before writing substantial content: target reader, excluded scope, central question, distinctive narrative shape, figure plan, and assessment focus.

2. Load context:
   - Read repository or project instructions when present;
   - Open the target document and preserve the user's existing structure unless it clearly harms the tutorial;
   - Open related series documents only to understand continuity, vocabulary, and quality expectations; choose the current chapter architecture from the topic's explanatory needs rather than a default formula;
   - For substantial work, style complaints, visual modes, or repeat iterations, read `references/tutorial-style-playbook.md`;
   - For diagram style work, AI/LLM tutorial figures, image-rich tutorials, or any `/tutorial-doc-style <mode>` request, also read `references/visual-style-baselines.md`.

3. Build evidence before writing:
   - Use web search when the topic depends on external technical facts;
   - Prefer a balanced source set: papers/arXiv, official docs, GitHub repositories or community docs, and high-quality blogs/tutorials;
   - Keep official facts, paper claims, community practice, and local interpretation clearly separated;
   - Verify framework/API capabilities against current official documentation when they may drift.

4. Plan the teaching path:
   - Start with enough context and narrative runway before naming the central problem;
   - Give a compact definition after the reader has a mental hook;
   - Move from intuition to mechanism, then to implementation details and decision rules;
   - Use natural transitions so the tutorial feels like one coherent article, not a sequence of figure captions;
   - Avoid formulaic series structure. Reuse a chapter sequence only when it is pedagogically justified for the current topic; otherwise vary the opening, mechanism order, examples, figure strategy, and decision framework according to the concept;
   - Treat central mechanisms as full teaching objects. Do not compress prerequisite reasoning, cost models, constraints, or failure modes into shallow asides;
   - Keep top-level teaching chapters to 6-7 or fewer, excluding references and learning assessment;
   - Keep headings short, formal, and durable.

5. Design figures as first-class teaching assets:
   - Select a visual mode from `references/visual-style-baselines.md`;
   - For substantial tutorials, create a figure plan with each figure's teaching purpose, chosen visual mode, required labels, and target workspace path before generating final assets;
   - If the user provides reference images, treat them as the canonical style target. If actual files are available, copy them into `assets/style-baselines/`; if they are only conversation images, encode their reusable visual rules in `references/visual-style-baselines.md`;
   - All named tutorial visual styles in this skill are `imagegen`-only for final figures. This includes `handdrawn`, `paper`, `dark-system`, and future style modes;
   - When the user asks for AI-era tutorial diagrams, bitmap tutorial figures, image-rich Markdown tutorials, or any named visual style from this skill, use `imagegen` as the primary generation path for the final visual asset;
   - Do not replace an `imagegen` request with Python/PIL, SVG, HTML/canvas, Mermaid, PlantUML, plotting libraries, scripts, or other deterministic rendering because of convenience, text-control concerns, or fear of label drift;
   - Do not use deterministic/local rendering as a post-processing or annotation layer for final tutorial images. If a generated figure has incorrect labels, cramped text, or layout problems, regenerate with a better `imagegen` prompt or split the figure;
   - If a user explicitly requests source-controlled vector/code-native diagrams, pause and clarify that this is a different deliverable mode outside this skill's final tutorial-image workflow. Do not silently switch the tutorial's figure workflow away from `imagegen`;
   - If exact formulas, matrix shapes, Chinese labels, or technical coordinates create risk for `imagegen`, solve it by prompt design, reference images, iterative generation, shortening labels, splitting figures, or asking the user to confirm a different visual strategy. Do not silently change the generation medium;
   - Clip hatching, fills, highlights, and strokes to their true semantic object. Never let a matrix fill spill outside the actual matrix boundary;
   - Give figures generous internal padding and inter-object spacing. If text is cramped, shorten text, enlarge the canvas, or split the figure;
   - Review every figure for teaching purpose, factual labels, shape correctness, communication semantics, visual hierarchy, and style fidelity;
   - Store final project-bound images in the local image/assets folder before referencing them.

6. Write in the target document style:
   - Use Simplified Chinese unless the user asks otherwise;
   - Follow the target repository or publishing environment for frontmatter, heading numbering, callouts, and punctuation;
   - Make the note self-contained outside the current conversation;
   - Avoid process narration, excessive quotation marks, repetitive caption templates, and unnecessary blank lines around LaTeX;
   - Use Chinese for reader-helping labels and English for stable professional terms from the target domain, such as framework concepts, operators, APIs, model names, and paper terms.

7. Add learning assessment for substantial tutorials:
   - Add a fixed final chapter named `学习测评` after references;
   - Include at least 10 customized questions, mostly single-choice or multiple-choice;
   - Present all questions first, then answers and explanations;
   - Use plausible technical distractors. Do not use absurd or unrelated joke options;
   - Give richer explanations for difficult or trap questions.

8. Validate:
   - Verify local image paths exist;
   - Run `git diff --check -- <target-file>` for edited Markdown files;
   - Run the skill validator when this skill changes;
   - Before finalizing, explicitly check that required project-bound image assets exist, every generated figure is referenced from the target document, the Multi-Agent Quality Gate has passed or was blocked by an explicitly reported tool-policy/user-authorization issue, and the assessment has been reviewed.

## Multi-Agent Quality Gate

When sub-agent tools are available and the revision is substantial, run a quality gate after the main draft is ready.

This gate is required for substantial tutorials. If the current environment exposes sub-agent tools only after discovery, discover them. If sub-agent tools exist but tool policy says they can only be used after an explicit user request, ask the user for authorization and do not silently downgrade to a solo review.

1. Spawn these agents in parallel:
   - Content accuracy reviewer: factual accuracy, source support, terminology, conceptual order;
   - Reader experience reviewer: clarity, continuity, transitions, depth, and whether the article feels exceptional;
   - Visual quality reviewer: figure purpose, technical correctness, visual style fidelity, spacing, labels, and anti-patterns;
   - Assessment author: draft the `学习测评` chapter.

2. Ask reviewers to return score out of 100, top risks, concrete suggestions, and pass/fail.

3. After integrating assessment questions, spawn a separate assessment reviewer for correctness, coverage, ambiguity, and difficulty balance.

4. Treat the tutorial as passing only when content accuracy plus reader experience exceeds 190/200 and the visual reviewer has no blockers.

5. Treat the assessment as passing only when the assessment reviewer gives no correctness blockers.

## Style Refinement Loop

When the user critiques the output:

1. Classify feedback as structure, depth, tone, visual strategy, source standard, format, assessment, or examples;
2. Apply the feedback to the current artifact first;
3. If the feedback should become reusable, update `references/tutorial-style-playbook.md` or `references/visual-style-baselines.md`;
4. Keep this `SKILL.md` concise and put detailed evolving style rules in references.

## References

- `references/tutorial-style-playbook.md`: writing structure, reader experience, source standards, assessment standards, and durable user preferences;
- `references/visual-style-baselines.md`: visual modes, canonical style references, prompt rules, and diagram anti-patterns.
