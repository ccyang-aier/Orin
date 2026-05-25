---
name: "orin-tutorial-style"
description: "Write, revise, and iteratively improve source-backed Chinese tutorial notes in the C:\\AIWorks\\Orin knowledge base, especially AI/LLM engineering tutorials that need Obsidian Markdown frontmatter, numbered headings, stable long-term wording, high-quality imagegen diagrams, explicit source links, and user-feedback-driven style refinement."
---

# Orin Tutorial Style

## Core Workflow

Use this skill when creating or revising Orin tutorial notes, especially under `notes/`, when the user expects a polished long-term learning artifact rather than a one-off answer.

1. Act as the main research-and-writing agent:
   - Load `AGENTS.md` for Orin-wide writing and git conventions;
   - Open the target note and preserve any user-provided outline or existing structure;
   - If this skill is being refined, also read `references/tutorial-style-playbook.md`;

2. Build evidence before writing:
   - Use web search when the topic depends on external technical facts;
   - Prefer a balanced source set: papers/arXiv, official docs, GitHub repositories or community docs, and high-quality blogs/tutorials;
   - Keep official facts, paper claims, community practice, and local interpretation clearly separated;

3. Plan the teaching path:
   - Start with enough context and narrative runway before naming the real problem;
   - Make the opening feel continuous, not like a sudden jump into an isolated question;
   - Give a compact definition after the reader has a mental hook;
   - Move from intuition to mechanism, then to implementation and decision rules;
   - Use explicit but natural transitions between sections so the tutorial feels like one continuous learning path;
   - Include common misconceptions and deployment/practice guidance when the topic is engineering-facing;
   - Keep top-level teaching chapters to 6-7 or fewer, excluding references and the learning assessment chapter;
   - Keep headings short, formal, and durable. Do not use casual headings or a heading as a sentence-length summary;

4. Use AI-era visuals deliberately:
   - Prefer generated raster diagrams via the `imagegen` skill for spatial, architectural, conceptual, or comparison-heavy explanations;
   - Copy final project-bound images into the note's local `imgs/` folder before referencing them;
   - Give every figure nearby guidance, but vary the form. Do not force every figure into the same "图 X 读图顺序" template;
   - Let figures serve the surrounding argument. The prose should not feel like it exists only to explain figures;
   - Use Chinese for reader-helping explanatory labels, and keep stable professional terms in English where they are standard;
   - Add enough figures for the length and conceptual density of the tutorial, but only when each figure clarifies a specific nearby knowledge point;
   - Verify generated figure labels for technical accuracy, not only visual quality. If a figure encodes the wrong mechanism, regenerate or replace it;
   - Use Mermaid/PlantUML only when exact syntax, deterministic graph text, or source-controlled diagram diffability matters more than visual quality;
   - Do not leave generated images only under `$CODEX_HOME/generated_images`;

5. Write in Orin tutorial style:
   - Use Simplified Chinese unless the user asks otherwise;
   - Add Obsidian YAML frontmatter with `tags`, `updated`, and `description`;
   - Use numbered H2/H3 headings such as `## 1. ...` and `### 1.1 ...`;
   - Avoid process narration such as "本次检索" or "刚才生成";
   - Make the note self-contained outside the current conversation;
   - Use concise tables where they sharpen contrast, not as decoration;
   - Avoid excessive quotation marks for emphasis. Prefer plain prose, bold text, or brackets when emphasis is truly needed;
   - Keep LaTeX display formulas close to the surrounding sentence; avoid unnecessary blank-line padding around formulas;
   - End list items with `；` when following Orin list punctuation conventions;

6. Add a learning assessment:
   - Add a fixed final chapter named `学习测评` after references;
   - Include at least 10 customized questions, mostly single-choice or multiple-choice;
   - Cover core concepts, transfer questions, pitfalls, difficult details, and deeper implementation understanding;
   - Present all questions first, then answers and explanations;

7. Close with sources and validation:
   - Add a final reference section with links actually used;
   - List references in one ordered list. Do not split references by source category unless the user asks for it;
   - Verify image paths exist and are repo-local;
   - Run `git diff --check -- <target-file>` on edited Markdown files;
   - Run `quick_validate.py` when this skill itself changes;

## Multi-Agent Quality Gate

After the main agent completes a substantial draft or revision, run a multi-agent quality gate when sub-agent tools are available.

1. Spawn these agents in parallel after the main content draft is ready:
   - Content accuracy reviewer: verify factual accuracy, source support, conceptual order, terminology, and whether the tutorial could mislead readers;
   - Reader experience reviewer: simulate a serious learner reading the note, judge clarity, depth, figure usefulness, continuity, transitions, narrative smoothness, and whether the tutorial feels exceptional rather than merely acceptable;
   - Assessment author: draft the `学习测评` chapter based on the tutorial content;

2. Ask each reviewer to return:
   - score out of 100;
   - top risks;
   - concrete improvement suggestions;
   - pass/fail judgment;

3. Ask the assessment author to return questions, choices, answers, and explanations. After integrating the assessment, spawn a separate assessment reviewer to verify question correctness, answer accuracy, coverage, difficulty balance, and whether any question is ambiguous.

4. Treat the main tutorial draft as passing only when the content accuracy reviewer plus reader experience reviewer combined score is greater than 190 out of 200.

5. Treat the assessment as passing only when the assessment reviewer explicitly passes it or gives no correctness blockers.

6. If the main score is 190 or lower, or the assessment reviewer finds blockers, revise using the highest-signal suggestions and repeat the relevant review loop.

7. Do not leak the intended answer to reviewers. Pass the note, relevant skill, and source requirements; ask for independent judgment.

## Style Refinement Loop

When the user critiques the output, treat the critique as training signal for this skill.

1. Identify whether the feedback is about:
   - structure;
   - depth;
   - tone;
   - visual strategy;
   - source standards;
   - Markdown/Obsidian format;
   - examples and analogies;

2. Apply the feedback to the current note first.

3. If the user explicitly wants the preference to become reusable, update `references/tutorial-style-playbook.md` with a short durable rule and, if needed, tighten this `SKILL.md`.

4. Keep the skill concise. Put detailed evolving style rules in the reference file, not in the main skill body.

## Reference

Read `references/tutorial-style-playbook.md` whenever the task is a substantial tutorial, a rewrite after user dissatisfaction, or a new note where style fidelity matters.
