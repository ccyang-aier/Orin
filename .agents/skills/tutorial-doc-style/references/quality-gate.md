# Quality Gate

Use this file when running the Multi-Agent Quality Gate from `SKILL.md`. The gate is reader-blocking first: a serious reader-experience blocker fails the tutorial even if numeric scores are high.

## 1. Review Packet

Before spawning any reviewer, provide:

- Confirmed teaching brief, including target reader, boundaries, exclusions, depth, content outline, figure plan, and assessment focus;
- Default reader when the brief is silent: an undergraduate with basic cognitive and derivation ability who is encountering the field for the first time;
- Full target document, not only excerpts;
- Image list with paths, captions, intended teaching purpose, and nearby prose;
- Source notes or citation list when factual claims depend on external evidence;
- The universal blockers below and the role-specific prompt.

## 2. Universal Reviewer Rules

- Read the document from beginning to end as a real learner; do not skim for isolated facts;
- Judge whether each concept, formula, figure, and conclusion has been earned by prior setup;
- Report section or line-level findings wherever possible;
- Treat a blocker as fail even when the document is otherwise accurate or polished;
- Do not give a pass because the topic is difficult, long, or visually rich.

Universal blockers:

- The lead-in abruptly introduces unexplained terms, formulas, tools, or subtopics before the reader has a reason to care;
- The prose says "back to", "as above", "therefore", or similar transitions when the prerequisite context was not actually established;
- A figure introduces objects, formulas, labels, or claims that the nearby prose has not prepared or explained;
- A key conclusion appears as a checklist or compressed claim without reasoning, example, or memorable anchor;
- Sections feel like a collection of knowledge points rather than a continuous teaching path;
- Assessment questions ask about article boundaries, author decisions, excluded topics, or process choices instead of knowledge the reader should understand;
- Assessment questions are absurd, jokey, trivial, ambiguous, or unrelated to the stated learning goals.

## 3. Role Prompts

Content accuracy reviewer:

```text
Review the full tutorial for factual accuracy, source support, terminology, and conceptual order. Check whether claims are supported by the provided sources or clearly framed as local explanation. Flag any factual error, unsupported drift-prone claim, misleading simplification, or concept introduced in an order that would confuse the target reader.
```

Reader experience reviewer:

```text
Review the full tutorial as an undergraduate with basic cognitive and derivation ability who is encountering this field for the first time, unless the teaching brief defines a different target reader. Check whether the article feels welcoming, whether the lead-in earns the topic, whether formulas and examples are introduced before being reused, whether transitions are logically valid, whether figures and prose teach together, and whether each major section leaves a durable memory point. Fail the review for any reader-blocking issue.
```

Visual quality reviewer:

```text
Review every figure using the image list, captions, purposes, and nearby prose. Check whether each figure has a clear teaching job, whether labels and objects are technically correct, whether the prose prepares and explains the figure, whether the layout is readable, and whether any visual anti-pattern appears. If image files are unavailable, report that limitation and still review figure intent and prose integration.
```

Assessment author:

```text
Draft the Learning Assessment only after reading the full tutorial and teaching brief. Questions must test knowledge understanding, transfer judgment, common misconceptions, or mechanism reasoning. Do not ask about why the article excludes a topic, how the author designed the tutorial, or any other meta writing decision. Mostly use single-choice and multiple-choice questions with plausible distractors.
```

Assessment reviewer:

```text
Review the final Learning Assessment for correctness, coverage, ambiguity, difficulty balance, and learning value. Reject any question that is a meta question about article scope or author decisions, any question unrelated to reader knowledge, any trivial or jokey distractor, and any answer explanation that does not teach the distinction being tested.
```

## 4. Output Schema

Each reviewer returns:

```markdown
Score: <0-100>
Pass: <yes/no>
Blocking issues:
- <section/line>: <issue and why it blocks the target reader>
Line/section findings:
- <section/line>: <specific problem>
Required revisions:
- <specific change required before pass>
Residual risks:
- <non-blocking concern, or "None">
```

The main agent must revise all blocking issues and rerun the failed review role before finalizing.
