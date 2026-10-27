# Provenance of the original s2 evaluation

The 1,540 `s2_*` runs in this folder are the **original evaluation** of the
frozen s2 protocol. They are kept exactly as uploaded, including every failed
or collapsed seed. Nothing made later may be mixed into them; new runs go to
other folders (e.g. `results/diagnostic_v1/`).

| Item | Value |
|---|---|
| Frozen commit | `998f2adb1c6cb244da311b3837319276d480d8f3` (local tag `s2-frozen`; the tag could not be pushed from the cloud session, so the commit hash is the reference) |
| Code reviewed externally | `012190e8decc5113eff0af2a945c2ac42036d5a8` (same frozen inputs and simulator) |
| Simulator fingerprint (`afl.provenance.code_hash`) | `fd90310c1b7c34e9` |
| Frozen manifest `configs/s2_frozen.json` SHA-256 | `8d04e337909babbad5e61cf319a1a66384cd993922ef40be34a38ab9d7640ffe` |
| Runs | 1,540 (s2_cifar 460, s2_femnist 460, s2_churn_cifar 80, s2_churn_femnist 80, s2_main_cifar 230, s2_main_femnist 230); 308,000 round records |
| Where they ran | Kaggle GPU, 4 parallel jobs, 37,564 s wall (`sweep.log`: 1540 run, 0 failed) |

## Uploaded archives (SHA-256)

| Archive | SHA-256 |
|---|---|
| `s2_cifar.zip` | `535e513a4c820391dbc205525fb3eed4f30e33e9fe9c78c20f4ec89c72f46858` |
| `s2_femnist.zip` | `1e5ec678b4b530d2b6420de792eaedee3b33a30b140a2769e65a2f891cb373c3` |
| `s2_churn_cifar.zip` | `cb703c3251b9e66160644427745550ffd80ee888dd767cfafa07d6fe6713c91a` |
| `s2_churn_femnist.zip` | `2441e366b2c664c58b8f5f086b0450a03276c7e5cd4c86ec490222778243ecab` |
| `s2_main_cifar.zip` | `7bc4ab836fd1ff79dc1f174d927b05660256ad9560d60b08508efa6bedfbe4df` |
| `s2_main_femnist.zip` | `12b6ed0a278fd4d10eb1bd2294d7655bb33ac3fd7bc981f3cff7aa2d4b682139` |

An external audit cited a combined archive `adv_results.zip` with SHA-256
`c78c8a2079aaeaf22f6588bb4329ee0798cae33ee6fa95f6556b67e8c3bb31d4`. That file
was never uploaded here, so it cannot be matched to the six archives above;
it was presumably re-zipped locally. The audit's 58 test results agree with
the table below, which is the stronger check.

## How to validate

    bash scripts/validate_s2_original.sh

It checks out the frozen commit (or the `s2-frozen` tag, if present) in a temporary worktree, points
`/kaggle/working/adv` (the output path recorded in every saved config) at that
checkout's `results/adv` with a symlink (never replacing a real directory),
runs the frozen `scripts/prespec_s2.py`, and compares its output with
`s2_prespecified_tests.md`, then runs the frozen code's own test suite in that
checkout. Expected: `OK: 58 results ... match` and `OK: frozen test suite: 153 passed`.
The validator's verdict-logic tests are skipped in the current tree (its
pinned inputs changed) and run here instead.

The current tree is **expected** to fail the manifest check once any pinned
file changes (for example the resume-validation repair to `scripts/sweep.py`).
That is by design: the frozen pins are never re-generated to accept new code.
