# Role Analysis Optimization Handoff

Last updated: 2026-04-28

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

## Recent Findings (2026-04-28 session)

### Speaker Confidence Feature

Added a per-segment speaker confidence token to the compact output format:

```
SEGS:
L1 0 ne h
L2 1 se m
L3 ? ne l
```

The fourth column is `h` (high), `m` (medium), or `l` (low). `?` speaker must always be `l`.

Implementation details:

- `_CONFIDENCE_CODES = {"h": "high", "m": "medium", "l": "low"}` in `role_analyzer.py`
- `_decode_compact_output` uses `split(None, 4)` and reads token[3] as confidence. Backward compatible: if token[3] is not a confidence code, it is treated as the anchor column (old format), and confidence defaults to `"high"`.
- Each decoded segment dict gains `_confidence` key.
- `sanitize_segments` was fixed to preserve `_`-prefixed metadata keys (they were previously stripped because the function rebuilt segments from explicit fields only).
- Frontend renders confidence badges and applies CSS classes `segment-conf-low` / `segment-conf-medium` on the segment card; filter flags `seg-flag-conf-low` / `seg-flag-conf-medium` appear on the row badge.

Current observation: with the tightened "陷阱" prompt rules the model almost always outputs `h`. Genuine uncertainty cases (ambiguous dialogue turn) may need prompt tuning to surface `m`/`l` more often.

### Prompt Optimization — Explicit Attribution Traps

Three common attribution failure patterns were isolated by diagnostic testing (model answered all three correctly in isolation → confirmed the original prompt was missing explicit rules for them). Three "陷阱" sections were added at the top of `COMPACT_ANALYSIS_PROMPT`:

**陷阱A — 称谓反推 (Title inference)**
When a line contains a title such as "少爷/大人/殿下/阁下", the speaker is almost certainly NOT the person being addressed. The addressee is being talked to, not speaking.

**陷阱B — 后置叙述线索 (Post-hoc narration cue)**
When a narration line AFTER a quote explicitly attributes it (e.g., "村长大大地挥手，否定了堤格尔的猜测"), that narration determines who said the preceding quote. Do not assign based on dialogue turn order alone.

**陷阱C — 动作叙述行 (Action description lines)**
Lines without `「」` that describe physical action, inner monologue, or transitional narration must be `0=旁白` even if they end with `：` and introduce the next quote.

Three few-shot examples were also tested in `/tmp/test_prompt_v2.py` to verify the improvement. The few-shot version is available but was not merged into the production prompt; the trap-rule wording alone was sufficient.

### Accuracy Numbers After Prompt Revision

Benchmark script: `/tmp/test_prompt_v2.py`

| Sample | Metric | Before | After |
|--------|--------|--------|-------|
| Sample B key trap segments (5 lines) | Accuracy | ~80% (4/5) | 100% (5/5) |
| Sample A full (~26 lines) | Accuracy | ~86% | ~88.5% (23/26) |
| Sample B full (~11 key lines) | Accuracy | 8/11 | 11/11 |

One remaining hard failure in Sample A: `「不，不是那边的邻居。」` is still attributed to 堤格尔 instead of 村长. The narration cue "村长大大地挥手" appears 3+ lines later; the trap-B rule currently only catches adjacent cases. Fixing this would require the model to look further ahead, or a two-pass approach.

### Bug Fixes (this session)

- **`sanitize_segments` dropping `_confidence`**: The function rebuilt segment dicts from explicit fields only, silently dropping any `_`-prefixed key. Fixed by appending a loop to copy all `_k` metadata keys.
- **Task snapshot resume starting from 0**: Imported `task_snapshot` type was routed to `runParseOnly` instead of `runAnalysisFlow`. `handleTaskSnapshotImport` now converts the snapshot to `analysis_flow` format, injecting `pipelineStage`/`pipelineIndex`/`pipelineCombo` from `_inMemoryLatestSnapshot` so the pipeline resumes from the correct chunk.
- **Segment doubling on resume**: `runRoleAnalysis` always initialized `segmentsState = segmentsState.slice()` at `startIndex=0`, appending to whatever was already in state. Fixed with `startIndex > 0` guard: `segments = startIndex > 0 ? segmentsState.slice() : []`.
- **Snapshot chooser dialog**: When both an imported snapshot and a live in-flight snapshot exist, `continueAnalysis()` now shows a chooser dialog instead of silently preferring one. Both entries display their timestamp and a short description.
- **QuotaExceededError toast spam**: `saveParseSnapshot` showed a toast on every chunk once localStorage was full. Fixed with `_snapshotQuotaWarned` one-shot flag, reset at `beginTask`.
- **beforeunload warning**: A `beforeunload` handler now warns when `taskBusy` is true or when `_snapshotQuotaWarned && inMemoryData` (data exists only in memory and would be lost on reload).
- **Danger button "清空文本与缓存"**: `clearTextAndCache()` function + button added below the text input to clear text, both snapshot slots, and all related localStorage keys.
- **`buildAnalysisFlowSnapshot` bloat**: Deduplicated `directAnalysisChunks` vs `chapterChunksState` when identical to avoid doubling snapshot size.

### Biggest Remaining Quality Issue

First-pass speaker reasoning in complex multi-person dialogue (tavern/group scenes). The prompt fixes reduce the most systematic errors but the model can still mis-assign a line in a fast-alternating 3-person exchange where no explicit speech tag is present.

Speaker verification remains off by default. Testing showed it introduced more over-corrections than it fixed.

## Suggested Next Steps

1. **Confidence calibration**: The tightened prompt makes the model output `h` for almost everything. Add a prompt instruction that `m` is appropriate when the speaker is inferred only from dialogue turn order (no explicit tag), and `l` when the speaker is genuinely ambiguous. This would make the confidence badges more actionable.

2. **Trap B — extended lookahead**: The trap-B rule (post-hoc narration cue) fails when the cue appears 3+ lines after the quote it attributes. A potential fix is a two-pass compact decode: first pass assigns speakers; second pass scans narration cues and back-propagates to the immediately preceding dialogue line.

3. **Build a hand-labeled benchmark for 2-4 representative scenes**. Include expected `speaker` only at first. Automate scoring with a script similar to `/tmp/test_prompt_v2.py` against the backend API.

4. **Per-chunk known character glossary**. Pass a glossary of character names seen in context so the model prefers complete forms (`拉菲纳克` over `菲纳克`).

5. **Revisit optional speaker verification only after there is a benchmark**. If kept, make it extremely conservative: only suggest changes for quote-only dialogue segments; do not change non-quote narration into a character; allow `UNKNOWN` as a safer output.

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
