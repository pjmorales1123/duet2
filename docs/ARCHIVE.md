# Development archive

The release keeps the runnable application, published model bytes, compact demonstrated-model inputs, exact result summaries and the current submission assets. Historical slide versions, unsuccessful research branches, frame-heavy traces and one-off development helpers have been removed from the main checkout.

Nothing in the published research record was discarded. The annotated tag [development-snapshot-2026-09-14](https://github.com/jannissio/talos-ai-infra/tree/development-snapshot-2026-09-14) preserves the complete sanitized repository at `f8eacd68420d3d392384cd41f810db55105de212`: all original models, training inputs, protocols, successful and failed evaluations, source snapshots and earlier presentations.

To inspect an old result, use the tag in GitHub or clone it separately:

```powershell
git clone --depth 1 --branch development-snapshot-2026-09-14 https://github.com/jannissio/talos-ai-infra.git talos-development-archive
```

Check disk first: this historical checkout is approximately 2.5 GiB before Git storage and optional environments. Keep the 10 GiB reserve plus expected writes. Its dated handoff notes describe past priorities; the current submission scope is defined by the main README and evidence index.

Several frozen manifests intentionally retain their original relative paths and hashes. Summaries remain in the release; full trace replay or a verifier requiring every original package member must use this complete archive. Do not edit frozen models or relabel failures to make a partial checkout satisfy a historical manifest.

Private local recordings, logs, rejected presentation drafts and uncommitted development work are also preserved outside the active repository on the author's machine. They are not included in this public tag or release. Neutral commit metadata and credential exclusion apply to both public refs.
