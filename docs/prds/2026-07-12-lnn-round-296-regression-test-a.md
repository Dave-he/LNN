---
title: "PRD #10-137 — Regression Test After r295 Default Promotion (r296)"
round: 296
date: 2026-07-12
author: "Claude (r296 /loop session)"
status: "selected"
parent: "r295 in-cell default promotion to all 3 gate cells"
paper: "internal regression"
variant: "A"
---

> **Selected** (round 296, 2026-07-12): r295 promoted decorrelation
> default to all 3 gate cells. This round runs a comprehensive
> regression test to confirm no existing tests broke from the new
> default behavior. Goal: full pytest suite green, with a documented
> count of test files / test cases passing.

# PRD #10-137 — Regression Test

## 目标
1. Run the full pytest suite to verify the r295 default change does
   not break any existing tests.
2. Document the total number of tests passing.
3. If anything broke, identify the cause and propose a fix or revert.

## 验收标准
- H1: full pytest suite green (no failures, no errors).
- H2: ≥ 173 tests passing (we had 173 before r295; should grow or stay).
- H3: document the regression test count in the report.

## 实现难度
**S** (15min). Just run pytest.

## 风险
- If H1 fails: a test was depending on the old default (no
  decorrelation). Need to either fix the test or revert the change.