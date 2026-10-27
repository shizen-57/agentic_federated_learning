#!/usr/bin/env bash
# Validate the original s2 runs with the ORIGINAL code, exactly as frozen.
#
#   bash scripts/validate_s2_original.sh
#
# Later commits may change files the s2 manifest pins (scripts/sweep.py,
# src/afl, ...), so the current tree is expected to fail prespec_s2.py's
# manifest check. That is by design: the original results are only ever
# validated from a checkout of the `s2-frozen` tag (commit 998f2ad), never by re-pinning.
#
# The saved run configs record their Kaggle output path, /kaggle/working/adv,
# and the frozen validator compares it. So this script points that path at
# the frozen checkout's results/adv with a symlink. It never replaces a real
# directory there, and restores a previous symlink afterwards.
set -euo pipefail

TAG=s2-frozen
FROZEN_COMMIT=998f2adb1c6cb244da311b3837319276d480d8f3   # what the tag points at
REPO=$(git rev-parse --show-toplevel)
WT=${WT:-$(mktemp -d)/s2-frozen}
LINK=/kaggle/working/adv

# Use the tag if this clone has it, otherwise the commit it names.
REF=$FROZEN_COMMIT
if git -C "$REPO" rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    [ "$(git -C "$REPO" rev-parse "$TAG^{commit}")" = "$FROZEN_COMMIT" ] \
        || { echo "tag $TAG does not point at $FROZEN_COMMIT; refusing"; exit 1; }
fi
git -C "$REPO" cat-file -e "$FROZEN_COMMIT^{commit}" 2>/dev/null \
    || { echo "frozen commit $FROZEN_COMMIT not in this clone (git fetch)"; exit 1; }
git -C "$REPO" worktree add --detach -q "$WT" "$REF"
old=""
if [ -L "$LINK" ]; then
    old=$(readlink "$LINK")
elif [ -e "$LINK" ]; then
    echo "$LINK exists and is not a symlink; refusing to touch it"; exit 1
fi
cleanup() {
    rm -f "$LINK"
    [ -n "$old" ] && ln -s "$old" "$LINK"
    git -C "$REPO" worktree remove --force "$WT"
}
trap cleanup EXIT
mkdir -p "$(dirname "$LINK")"
ln -sfn "$WT/results/adv" "$LINK"

cd "$WT"
out=$(mktemp)
{
    echo "# Pre-specified s2 tests (scripts/prespec_s2.py --root /kaggle/working/adv; see docs/ADVANCED_RESULTS.md)"
    python scripts/prespec_s2.py --root "$LINK"
} > "$out" || { cat "$out"; echo "FAILED: the frozen validator gave no verdict"; exit 1; }
if diff -q "$out" results/adv/s2_prespecified_tests.md >/dev/null; then
    echo "OK: $(grep -c '^| s2_' "$out") results from the frozen code match results/adv/s2_prespecified_tests.md"
else
    diff "$out" results/adv/s2_prespecified_tests.md || true
    echo "MISMATCH against the committed table"; exit 1
fi
