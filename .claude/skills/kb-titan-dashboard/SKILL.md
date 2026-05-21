---
name: kb-titan-dashboard
description: Update the kintera × KB Titan state/decision dashboard at /home/sam2/dev/kintera/dashboard/. Use whenever the user needs to make a decision about next steps, or after each significant experiment so they can see the current state. Regenerate figures, refresh HTML, and (re)start a local HTTP server so the user can view in browser via SSH port-forward.
---

# kb-titan-dashboard

A persistent decision dashboard for the kintera × KB Titan compatibility work. The user wants this updated every time they need to make a choice or see overall progress. Keep it terse, visual, and decision-oriented — not a report archive.

## When to invoke

- The user is about to decide a direction ("what's next?", "which option?", "should we keep going?")
- We just finished a phase / experiment that materially changes the state
- The user explicitly asks for a dashboard update

## When NOT to invoke

- Mid-experiment (regenerate after, not during)
- For small tweaks that don't change the headline metrics or open the next-step set

## Workflow

1. **Confirm current best dump exists** — typically `/tmp/baseline_g29.npz` or whatever the latest accepted G-config produces. If the baseline has shifted, re-run G29 first to refresh.

2. **Edit `/home/sam2/dev/kintera/dashboard/make_figures.py`** to add / remove / refresh figures. Keep the figure count modest (10-12 max); replace stale figs rather than appending. Each figure should answer a specific decision question.

3. **Regenerate figures**:
   ```bash
   /opt/anaconda3/bin/python /home/sam2/dev/kintera/dashboard/make_figures.py
   ```
   Should print `[done] generated N figures in ...`.

4. **Edit `/home/sam2/dev/kintera/dashboard/index.html`** to reflect any new sections, updated headline metrics, the current best-config string, branch+commit hash, and the next-step decision matrix. Keep nav links + summary boxes in sync.

5. **Start/restart the HTTP server**:
   ```bash
   pkill -f "python -m http.server 8000" 2>/dev/null || true
   cd /home/sam2/dev/kintera/dashboard
   nohup /opt/anaconda3/bin/python -m http.server 8000 --bind 0.0.0.0 > /dev/null 2>&1 &
   ```

6. **Tell the user how to view**:
   ```
   ssh -L 8000:localhost:8000 <host>
   # then open http://localhost:8000/index.html
   ```

7. **Summarize in chat**: 2-3 sentences naming the new figures, the recommended next step, and any new red-flag findings from the regenerated data.

## Standard sections to maintain in `index.html`

- **§1 Headline metrics** — 4-tile stat grid (matched count, match rates, key ratio)
- **§2 Refactor state** — what landed, what's pending, link to REFACTOR_SCHEMA.html
- **§3 Match analysis** — heatmap, ratio histogram, % matched vs altitude
- **§4 Profiles** — vertical curves for top species
- **§5 Cation/grain snapshots** — bar charts at the relevant altitude
- **§6 Remaining gaps** — visualized causal traces (Fig 9 cation bleed, Fig 8 grain) with concise text
- **§7 What we've explored** — sweeps, attempted fixes, negative results — so we don't redo them
- **§8 Next-step options** — 4-option matrix with effort/risk/reward + recommendation

## Important details

- **Inline base64 vs separate PNGs**: keep PNGs as separate files in `figs/`. Smaller HTML, easier to spot-check individual figures during iteration.
- **Don't auto-push** the dashboard files. They're working memory; the user pushes when they want.
- **Preserve the user's "last seen" mental model**: when something material changes, call it out explicitly in chat ("you saw 159 matched last time; we're at X now because…").
- **Figure data sources** should be readable from `/tmp/*.npz` dumps — don't hard-code numbers in the figure script if a dump is available.
- **Server runs in background**; assume the user keeps the SSH tunnel open during a session. Don't restart it on every minor change.

## Pitfalls to avoid

- Don't accumulate stale figures. If a figure no longer pulls weight, delete the matplotlib block and remove from HTML.
- Don't write a "report" version. The dashboard is for decision support, not a paper. Sentences in HTML should be ≤2 lines.
- Don't visualize what's already obvious from a number — a single stat tile beats a one-bar chart.
- Don't generate a figure without describing what decision it informs in its caption.

## Quick smoke test after each update

```bash
ls /home/sam2/dev/kintera/dashboard/figs/   # verify all PNGs present
ss -ltnp | grep :8000                        # verify server alive
```

If the server died (no listener on 8000), restart per step 5.
