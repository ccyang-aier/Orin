---
name: "tutorial-doc-style"
description: "Write, revise, and iteratively improve source-backed technical tutorial documents, especially Chinese AI/LLM engineering tutorials that need clear narrative structure, rigorous concepts, high-quality educational diagrams, professional learning assessments, explicit sources, and user-feedback-driven style refinement. Use for Markdown notes, knowledge-base articles, repo docs, or long-form tutorials; adapt to the target project's conventions instead of assuming any specific workspace or editor."
---

# Tutorial Document Style

Use this skill when a tutorial should become a polished long-term learning artifact rather than a one-off answer.

## Workflow

1. Load context:
   - Read repository or project instructions when present;
   - Open the target document and preserve the user's existing structure unless it clearly harms the tutorial;
   - For substantial work, style complaints, visual modes, or repeat iterations, read `references/tutorial-style-playbook.md`;
   - For diagram style work or any `/tutorial-doc-style <mode>` request, also read `references/visual-style-baselines.md`.

2. Build evidence before writing:
   - Use web search when the topic depends on external technical facts;
   - Prefer a balanced source set: papers/arXiv, official docs, GitHub repositories or community docs, and high-quality blogs/tutorials;
   - Keep official facts, paper claims, community practice, and local interpretation clearly separated;
   - Verify framework/API capabilities against current official documentation when they may drift.

3. Plan the teaching path:
   - Start with enough context and narrative runway before naming the central problem;
   - Give a compact definition after the reader has a mental hook;
   - Move from intuition to mechanism, then to implementation details and decision rules;
   - Use natural transitions so the tutorial feels like one coherent article, not a sequence of figure captions;
   - Treat central mechanisms as full teaching objects. Do not compress important topics such as communication cost into a shallow aside;
   - Keep top-level teaching chapters to 6-7 or fewer, excluding references and learning assessment;
   - Keep headings short, formal, and durable.

4. Design figures as first-class teaching assets:
   - Select a visual mode from `references/visual-style-baselines.md`;
   - If the user provides reference images, treat them as the canonical style target. If actual files are available, copy them into `assets/style-baselines/`; if they are only conversation images, encode their reusable visual rules in `references/visual-style-baselines.md`;
   - All named tutorial visual styles in this skill are `imagegen`-first. This includes `handdrawn`, `paper`, `paper-detailed`, `dark-system`, and future style modes unless the user explicitly says otherwise;
   - When the user asks for AI-era tutorial diagrams, bitmap tutorial figures, image-rich Markdown tutorials, or any named visual style from this skill, use `imagegen` as the primary generation path for the final visual asset;
   - Do not replace an `imagegen` request with Python/PIL, SVG, HTML/canvas, Mermaid, PlantUML, or other deterministic rendering because of convenience or fear of label drift;
   - Deterministic rendering may be used only when the user explicitly asks for code-native diagrams, when the target format requires source-controlled vector output, or as a clearly disclosed post-processing/annotation aid after an `imagegen` draft exists;
   - If exact formulas, matrix shapes, Chinese labels, or technical coordinates create risk for `imagegen`, solve it by prompt design, reference images, iterative generation, selective post-processing, or asking for confirmation. Do not silently change the generation medium;
   - Clip hatching, fills, highlights, and strokes to their true semantic object. Never let a matrix fill spill outside the actual matrix boundary;
   - Give figures generous internal padding and inter-object spacing. If text is cramped, shorten text, enlarge the canvas, or split the figure;
   - Review every figure for teaching purpose, factual labels, shape correctness, communication semantics, visual hierarchy, and style fidelity;
   - Store final project-bound images in the local image/assets folder before referencing them.

5. Write in the target document style:
   - Use Simplified Chinese unless the user asks otherwise;
   - Follow the target repository or publishing environment for frontmatter, heading numbering, callouts, and punctuation;
   - Make the note self-contained outside the current conversation;
   - Avoid process narration, excessive quotation marks, repetitive caption templates, and unnecessary blank lines around LaTeX;
   - Use Chinese for reader-helping labels and English for stable professional terms such as `Tensor Parallelism`, `all-reduce`, `hidden states`, and `Column Parallel`.

6. Add learning assessment for substantial tutorials:
   - Add a fixed final chapter named `学习测评` after references;
   - Include at least 10 customized questions, mostly single-choice or multiple-choice;
   - Present all questions first, then answers and explanations;
   - Use plausible technical distractors. Do not use absurd or unrelated joke options;
   - Give richer explanations for difficult or trap questions.

7. Validate:
   - Verify local image paths exist;
   - Run `git diff --check -- <target-file>` for edited Markdown files;
   - Run the skill validator when this skill changes.

## Multi-Agent Quality Gate

When sub-agent tools are available and the revision is substantial, run a quality gate after the main draft is ready.

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
