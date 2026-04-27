# Role Analysis Optimization Handoff

Last updated: 2026-04-27

This document records the current state of the role/dialogue analysis optimization work so another AI or developer can continue without replaying the full chat history.

## Current Direction

The current preferred strategy is:

- Use compact output mode for role analysis to reduce output tokens.
- Keep the first-pass analysis as the main source of truth.
- Do not use mechanical speaker post-processing.
- Keep only metadata cleanup and style stabilization after model output.
- Keep `UNKNOWN` as `UNKNOWN` for manual review instead of converting it to narration.
- Keep speaker verification available as an optional manual switch, but default it off because recent tests showed over-correction.

Avoid reverting to automatic compact-to-verbose fallback. It was tested and tends to make failures worse by expanding output and causing `finish_reason=length`.

## Main Code Changes

Changed files:

- `role_analyzer.py`
- `llm_client.py`
- `schemas.py`
- `static/index.html`

Important behavior now:

- `LLMConfig.output_mode` defaults to `compact`.
- `LLMConfig.enable_speaker_verification` defaults to `False`.
- Frontend speaker verification checkbox is unchecked by default.
- Frontend compact output checkbox is checked by default.
- Continuing analysis can choose current page config or snapshot config through the existing single continue button flow.
- `UNKNOWN` speakers are preserved and marked with `_needs_review`.
- `UNKNOWN` and `旁白` styles are stabilized to `female, moderate pitch`.
- Known character gender style stabilization remains active.
- `postprocess_segments()` only does metadata marking and style stabilization; it does not alter speaker/text segmentation.

## Things Deliberately Removed Or Avoided

Do not reintroduce these without a strong reason:

- Regex-based splitting of `「对白」某某说道`.
- Regex-based conversion of action narration to `旁白`.
- Neighbor-based inference from narration to rewrite quote speaker.
- Merging expression quote snippets such as `露出「真伤脑筋」的表情`.
- Automatic fallback from compact mode to verbose JSON mode.
- Automatic `UNKNOWN -> 旁白`.

These approaches looked helpful in isolated cases but caused false positives and hid uncertainty that should remain visible for review.

## Current Prompt Strategy

The compact prompt currently emphasizes:

- Non-quote lines default to `0=旁白`, except clear inner monologue or direct character speech.
- Quote text is not always dialogue; titles, rumors, metaphors, and quoted phrases in narration should remain narration.
- Speaker should be inferred from explicit speech tags and dialogue turn state, not just nearest character name.
- Character names are proper nouns. Prefer complete names seen in source text; do not semantically or visually truncate a name such as `拉菲纳克` into `菲纳克` unless the source explicitly uses that shorter form.
- Style must be short whitelist tags only, not personality descriptions.

This last style restriction was necessary because `qwen/qwen3.6-27b` once produced an extremely long `ROLES` style list and hit `finish_reason=length`.

## Retry And Failure Behavior

`chat_json()` already had retry support.

`chat_text()`/compact paths now also go through `_run_chat_text_with_retry()` in `role_analyzer.py`:

- Retries transient errors such as timeout, disconnection, empty response, 429, and 5xx.
- Does not blindly retry `finish_reason=length`, because that is usually deterministic capacity overflow.
- Compact role analysis retries a decode failure on the same chunk once before splitting smaller.
- If compact still fails on a minimal chunk, it raises a compact-format error instead of falling back to verbose JSON.

`llm_client.py` now checks `finish_reason=length` in `chat_text()` too, not just `chat_json()`.

## Local Model Test Setup

Local OpenAI-compatible endpoint:

- Base URL: `http://127.0.0.1:1234`
- API key: `local`
- Compatibility mode: `chat_compat`
- Output mode: `compact`
- Speaker verification: off for quality tests
- Temperature used in latest tests: `0.08`
- Max tokens: `8000`
- Workers: `1`

Source text used:

- `/Users/coolking70/Documents/同步空间/aitts/text/2546_简体统一修复版-第二卷+.txt`

Representative sample markers:

- Sample A: `这话题告一段落后，村长问起两人今后准备前往何方。`, about 2600 chars.
- Sample B: `隔天早上，堤格尔与拉菲纳克一起进入森林。`, about 3200 chars.

## Model Test Results

### qwen/qwen3.6-35b-a3b

Using current style-constrained compact prompt:

- Sample A: about 60 segments, approximate accuracy `86%-88%`.
- Sample B: about 59 segments, approximate accuracy `96%-98%`.

Observations:

- Sample B is strong and mostly usable.
- Sample A still has multi-speaker tavern dialogue issues.
- Common issues: occasional speaker turn mistakes, occasional shortened name `菲纳克`, and some prompt/narration lines still assigned to a character.

### qwen/qwen3.6-27b

Using current style-constrained compact prompt:

- Sample A: 61 segments, approximate accuracy `86%-89%`.
- Sample B: 59 segments, approximate accuracy `96%-98%`.

Observations:

- Quality is close to 35B and Sample A may be slightly better in a few places.
- Runtime was much slower in the latest test, over 10 minutes for two samples.
- It still produced one `菲纳克` shortened-name occurrence.
- It had a few Sample A speaker mistakes, including assigning some lines to the wrong speaker in tavern dialogue.

## Recent Findings

The biggest remaining quality issue is not output format stability anymore. It is first-pass speaker reasoning in complex multi-person dialogue.

Current high-value error types:

- Multiple active speakers in a tavern/group scene.
- Shortened or partial proper names, especially `菲纳克` vs `拉菲纳克`.
- Non-quote speech-tag lines occasionally assigned to characters despite prompt saying default narration.
- A few dialogue turns after narration insertions can still be offset by one speaker.

Speaker verification was tested and currently should remain off by default. It fixed some lines but introduced over-corrections, including changing narration/action segments into character speech.

## Suggested Next Steps

1. Build a small hand-labeled benchmark for 2-4 representative scenes.

   Include expected `speaker` only at first. Do not try to judge emotion/style yet.

2. Add an automated scoring script.

   Feed the same sample through the model, compare segment-level speaker accuracy, and report mismatch lines.

3. Improve compact prompt with few-shot examples.

   Focus examples on:
   - Multi-person tavern dialogue.
   - `某某说道/问道` prompt lines as narration.
   - Proper-noun preservation.
   - Non-dialogue quoted phrases.

4. Consider passing a per-chunk known character glossary.

   This should guide names without forcing alias merge. The wording should be "prefer complete names observed in source/context" rather than "merge similar names".

5. Revisit optional speaker verification only after there is a benchmark.

   If kept, make it extremely conservative:
   - Only suggest changes for quote-only dialogue segments.
   - Do not change non-quote narration into a character.
   - Allow `UNKNOWN` as a safer output.

## Quick Verification Commands

Run syntax and whitespace checks:

```bash
python -m py_compile role_analyzer.py llm_client.py schemas.py app.py
git diff --check
```

Check frontend inline script parses:

```bash
node - <<'NODE'
const fs = require('fs');
const html = fs.readFileSync('static/index.html', 'utf8');
const scripts = [...html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/gi)].map(m => m[1]);
for (const script of scripts) new Function(script);
console.log(`parsed ${scripts.length} inline scripts`);
NODE
```

## Worktree Notes

At the time this handoff was created, the worktree had modified files and some untracked sample files. Do not assume a clean tree.

Known untracked files seen earlier:

- `samples/第 01 章.md`
- `samples/第 02 章.md`
- `samples/第 03 章.md`

Before committing or handing off, review `git status --short`.
