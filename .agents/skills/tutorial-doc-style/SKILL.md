---
name: "tutorial-doc-style"
description: "Write, revise, and iteratively improve source-backed technical tutorial documents, especially Chinese AI/LLM engineering tutorials that need clear narrative structure, rigorous concepts, high-quality educational diagrams, professional assessments, explicit sources, and iterative user-feedback-driven style refinement. Use for Markdown notes, knowledge-base articles, repo docs, or long-form tutorials; adapt to the target project's conventions instead of assuming any specific workspace or editor."
---

# Tutorial Document Style

## Core Workflow

Use this skill when creating or revising a tutorial document that should become a polished long-term learning artifact rather than a one-off answer.

1. Act as the main research-and-writing agent:
   - Load local repository or project instructions when present;
   - Open the target note and preserve any user-provided outline or existing structure;
   - If this skill is being refined, also read `references/tutorial-style-playbook.md`;

2. Build evidence before writing:
   - Use web search when the topic depends on external technical facts;
   - Prefer a balanced source set: papers/arXiv, official docs, GitHub repositories or community docs, and high-quality blogs/tutorials;
   - Keep official facts, paper claims, community practice, and local interpretation clearly separated;
   - Verify framework/API capabilities against current official documentation, and state the version or context when the capability is likely to drift;

3. Plan the teaching path:
   - Start with enough context and narrative runway before naming the real problem;
   - Make the opening feel continuous, not like a sudden jump into an isolated question;
   - Give a compact definition after the reader has a mental hook;
   - Move from intuition to mechanism, then to implementation and decision rules;
   - Use explicit but natural transitions between sections so the tutorial feels like one continuous learning path;
   - Identify core knowledge points and give them enough depth. Do not compress a central mechanism into a shallow paragraph;
   - Include common misconceptions and deployment/practice guidance when the topic is engineering-facing;
   - Keep top-level teaching chapters to 6-7 or fewer, excluding references and the learning assessment chapter;
   - Keep headings short, formal, and durable. Do not use casual headings or a heading as a sentence-length summary;

4. Use AI-era visuals deliberately:
   - Prefer high-quality visual assets for spatial, architectural, conceptual, or comparison-heavy explanations;
   - Use `imagegen` for polished conceptual diagrams, architectural maps, and visually rich explainers;
   - Use deterministic SVG/HTML/canvas/Python-rendered figures when exact dimensions, formulas, labels, or coordinate relationships must be guaranteed;
   - External figures from papers, docs, or blogs may be used only when they are clearly sourced and materially better than a new diagram. If a source figure is close but not ideal, use it as inspiration and redraw or improve it instead of blindly screenshotting;
   - Copy final project-bound images into a local image/assets folder before referencing them;
   - Give every figure nearby guidance, but vary the form. Do not force every figure into the same "图 X 读图顺序" template;
   - Let figures serve the surrounding argument. The prose should not feel like it exists only to explain figures;
   - Use Chinese for reader-helping explanatory labels, and keep stable professional terms in English where they are standard;
   - Add enough figures for the length and conceptual density of the tutorial, but only when each figure clarifies a specific nearby knowledge point;
   - Review every figure against its intended teaching point: purpose, logic, technical accuracy, visual hierarchy, and whether the reader can understand it at a glance;
   - Verify generated figure labels for technical accuracy, not only visual quality. If a figure encodes the wrong mechanism, regenerate or replace it;
   - Use Mermaid/PlantUML only when exact syntax, deterministic graph text, or source-controlled diagram diffability matters more than visual quality;
   - Do not leave generated images only under `$CODEX_HOME/generated_images`;

5. Write in the target document style:
   - Use Simplified Chinese unless the user asks otherwise;
   - Follow the target repository, platform, or publishing environment. Add YAML frontmatter, numbered headings, callouts, or metadata only when the target context expects them;
   - Avoid process narration such as "本次检索" or "刚才生成";
   - Make the note self-contained outside the current conversation;
   - Use concise tables where they sharpen contrast, not as decoration;
   - Avoid excessive quotation marks for emphasis. Prefer plain prose, bold text, or brackets when emphasis is truly needed;
   - Keep LaTeX display formulas close to the surrounding sentence; avoid unnecessary blank-line padding around formulas;
   - Follow the target document's punctuation and list conventions;

6. Add a learning assessment:
   - Add a fixed final chapter named `学习测评` after references;
   - Include at least 10 customized questions, mostly single-choice or multiple-choice;
   - Cover core concepts, transfer questions, pitfalls, difficult details, and deeper implementation understanding;
   - Present all questions first, then answers and explanations;
   - Keep distractors professional. Wrong options should be plausible misconceptions or related traps, not irrelevant jokes or obviously absurd choices;
   - Give richer explanations for complex or difficult questions, especially when a wrong answer reveals a misunderstanding;

7. Close with sources and validation:
   - Add a final reference section with links actually used;
   - List references in one ordered list. Do not split references by source category unless the user asks for it;
   - Verify image paths exist and are local to the deliverable when the document references local assets;
   - Run `git diff --check -- <target-file>` on edited Markdown files;
   - Run `quick_validate.py` when this skill itself changes;

## Multi-Agent Quality Gate

After the main agent completes a substantial draft or revision, run a multi-agent quality gate when sub-agent tools are available.

1. Spawn these agents in parallel after the main content draft is ready:
   - Content accuracy reviewer: verify factual accuracy, source support, conceptual order, terminology, and whether the tutorial could mislead readers;
   - Reader experience reviewer: simulate a serious learner reading the note, judge clarity, depth, figure usefulness, continuity, transitions, narrative smoothness, and whether the tutorial feels exceptional rather than merely acceptable;
   - Visual quality reviewer: inspect whether figures clearly present their intended logic, use accurate labels, show necessary details, and feel professional enough for a high-quality technical tutorial;
   - Assessment author: draft the `学习测评` chapter based on the tutorial content;

2. Ask each reviewer to return:
   - score out of 100;
   - top risks;
   - concrete improvement suggestions;
   - pass/fail judgment;

3. Ask the assessment author to return questions, choices, answers, and explanations. After integrating the assessment, spawn a separate assessment reviewer to verify question correctness, answer accuracy, coverage, difficulty balance, and whether any question is ambiguous.

4. Treat the main tutorial draft as passing only when the content accuracy reviewer plus reader experience reviewer combined score is greater than 190 out of 200, and the visual reviewer has no blockers.

5. Treat the assessment as passing only when the assessment reviewer explicitly passes it or gives no correctness blockers.

6. If the main score is 190 or lower, the visual reviewer finds blockers, or the assessment reviewer finds blockers, revise using the highest-signal suggestions and repeat the relevant review loop.

7. Do not leak the intended answer to reviewers. Pass the note, relevant skill, and source requirements; ask for independent judgment.

## Style Refinement Loop

When the user critiques the output, treat the critique as training signal for this skill.

1. Identify whether the feedback is about:
   - structure;
   - depth;
   - tone;
   - visual strategy;
   - source standards;
   - Markdown or publishing format;
   - examples and analogies;

2. Apply the feedback to the current note first.

3. If the user explicitly wants the preference to become reusable, update `references/tutorial-style-playbook.md` with a short durable rule and, if needed, tighten this `SKILL.md`.

4. Keep the skill concise. Put detailed evolving style rules in the reference file, not in the main skill body.

## Reference

Read `references/tutorial-style-playbook.md` whenever the task is a substantial tutorial, a rewrite after user dissatisfaction, or a new note where style fidelity matters.
