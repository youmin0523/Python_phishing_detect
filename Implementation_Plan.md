# Implementation Plan - Phishing Detection Accuracy Improvement

## Phase 1: Confidence Logic Refactoring

### 1-1. `test.py` Modification

- **Current Logic**:
  - Uses linear approximation based on `decision_score` (0.5 + score/2).
  - Saturates at 1.0 quickly.
- **Improved Logic**:
  - Use Sigmoid function: `1 / (1 + exp(-|decision_score|))`.
  - Provides smooth transition from 0.5 to 1.0 for any `decision_score` range.
- **Dependency**:
  - Add `import math` (standard library).

### 1-2. Verification

- `test.py` should run without errors.
- Output confidence scores should reflect more granular probability values (e.g., instead of 1.0 or 0.5, something like 0.88 if `score` is 2.0).

## Phase 2: Testing

- Run `python test.py` to see the new output format.
