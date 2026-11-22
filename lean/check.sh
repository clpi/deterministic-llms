#!/usr/bin/env bash
# Cold build of the PRLM Lean package with warnings as errors, then the
# hygiene scans: no sorry/admit, no custom axiom, no escape hatches, and
# the axioms of every theorem printed.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.elan/bin:$PATH"
rm -rf .lake/build
lake build -R -KwarningAsError=true Prlm prlm 2>&1 | tail -3
bad=$(grep -nE '\b(sorry|admit|native_decide|implemented_by|extern|unsafe)\b|^axiom ' Prlm/*.lean Main.lean | grep -v -- '--' || true)
if [ -n "$bad" ]; then echo "HYGIENE FAIL"; echo "$bad"; exit 1; fi
cat > /tmp/prlm_axioms.lean <<'X'
import Prlm.Core
import Prlm.AlgoSearch
import Prlm.Skip
import Prlm.Sparse
import Prlm.Loop
import Prlm.Fix
import Prlm.Struct
import Prlm.Robust
open Prlm
#print axioms best_perm
#print axioms max_unique
#print axioms certify_sound
#print axioms certify_complete
#print axioms rescore_eq
#print axioms stable_of_bound
#print axioms scores_local
#print axioms dot_set_zero
#print axioms gen_prefix
#print axioms gen_perm
#print axioms gen_local
#print axioms gen_cache
#print axioms stepRe_eq
#print axioms gen_reweight
#print axioms accept_gen
#print axioms accept_batch
#print axioms best_split
#print axioms AlgoSearch.mem_arch
#print axioms AlgoSearch.cheb_target
#print axioms Prlm.Loop.arch_cons_sub
#print axioms Prlm.Loop.arch_covers
#print axioms Prlm.Loop.admit_sound
#print axioms Prlm.Loop.reject_sound
#print axioms Prlm.Loop.scope_sound
#print axioms Prlm.Fix.closed_iff_no_proposals
#print axioms Prlm.Fix.closed_local_opt
#print axioms Prlm.Fix.fix_stable
#print axioms Prlm.Fix.point_local_opt
#print axioms Prlm.Fix.coarse_le
#print axioms Prlm.Fix.coarse_front_within
#print axioms Prlm.Struct.wins_iff
#print axioms Prlm.Struct.minshift_wins
#print axioms Prlm.Struct.minshift_minimal
#print axioms Prlm.Struct.picked_iff
#print axioms Prlm.Struct.minshiftAll_wins
#print axioms Prlm.Struct.minshiftAll_minimal
#print axioms Prlm.Struct.rescore_comm
#print axioms Prlm.Struct.replan_local
#print axioms Prlm.Struct.parse_render
#print axioms Prlm.Struct.render_set
#print axioms Prlm.Robust.pair_iff
#print axioms Prlm.Robust.vocabulary_iff
#print axioms Prlm.Robust.radius_iff
#print axioms Prlm.Robust.interval_box_iff
#print axioms Prlm.Robust.relative_iff
#print axioms Prlm.Robust.uniform_drift_preserves
#print axioms Prlm.Robust.trace_eq_of_on_reference_prefix
X
ax=$(lake env lean /tmp/prlm_axioms.lean 2>&1) || { echo "$ax"; echo "AXIOM CHECK FAILED"; exit 1; }
echo "$ax" | grep -q "error" && { echo "$ax"; echo "AXIOM CHECK FAILED"; exit 1; }
echo "$ax" | grep -v "^'AlgoSearch"
echo "$ax" | grep -E "sorryAx|depends on axioms: \[.*[^]]*" | grep -vE "\[(propext|Classical.choice|Quot.sound)(, (propext|Classical.choice|Quot.sound))*\]" && { echo "NONSTANDARD AXIOM"; exit 1; } || true
echo "check: pass"
