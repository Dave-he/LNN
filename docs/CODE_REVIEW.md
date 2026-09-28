# 代码审查 (Code review)

Automated top-N observations. Not a substitute for human review.

**Status**: `active`

## Findings

1. **Large module**: `projects/llama.cpp/tools/ui/.svelte-kit/output/client/_app/immutable/bundle.bFnAB0IV.js` is 132157 LOC — split candidates above 500 LOC.
2. **HACK density**: 7 `HACK` markers — review for shortcuts.
3. **FIXME density**: 24 `FIXME` markers — promote to issues.

## Recommended human review checklist

- [ ] Read README/AGENTS.md to grasp intent
- [ ] Skim top 3 largest files for responsibility leaks
- [ ] Check error-handling strategy (typed errors vs exceptions vs Result)
- [ ] Verify config/secrets loading
- [ ] Confirm test fixtures cover the happy + error paths
