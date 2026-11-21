/-
  When the self-improving loop stops (tasks/FIXPOINT_CONTRACT.json).

  The loop evaluates parameter points `E` (objective `f : α → V`) and
  proposes the unit neighbours `N p` of every point `p` on the front of
  what it has evaluated. It stops when a round proposes nothing new.

  * closed_iff_no_proposals  a round proposes nothing exactly when every
                             neighbour of every front point was evaluated
  * closed_local_opt         then no neighbour of a front point dominates it:
                             the front is locally Pareto optimal for the
                             neighbourhood `N`, which is a certificate, not a
                             budget stop
  * fix_stable               and the next round changes nothing

  Termination itself rests on the grid being finite (each round that does
  not stop adds at least one unevaluated grid point); that counting
  argument is stated in the contract and is not formalized here.
-/
import Prlm.AlgoSearch

namespace Prlm.Fix
open AlgoSearch

variable {α : Type} [DecidableEq α]

/-- Every neighbour of every front point has been evaluated. -/
def Closed (f : α → V) (N : α → List α) (E : List α) : Prop :=
  ∀ p ∈ E, f p ∈ arch (E.map f) → ∀ q ∈ N p, q ∈ E

/-- What the next round would evaluate. -/
def proposals (f : α → V) (N : α → List α) (E : List α) : List α :=
  ((E.filter (fun p => decide (f p ∈ arch (E.map f)))).flatMap N).filter (fun q => decide (q ∉ E))

theorem closed_iff_no_proposals (f : α → V) (N : α → List α) (E : List α) :
    proposals f N E = [] ↔ Closed f N E := by
  simp only [proposals, Closed, List.filter_eq_nil_iff, List.mem_flatMap, List.mem_filter,
    decide_eq_true_eq, Decidable.not_not]
  constructor
  · intro h p hp hf q hq
    exact h q ⟨p, ⟨hp, hf⟩, hq⟩
  · rintro h q ⟨p, ⟨hp, hf⟩, hq⟩
    exact h p hp hf q hq

omit [DecidableEq α] in
theorem closed_local_opt (f : α → V) (N : α → List α) (E : List α)
    (hc : Closed f N E) (p : α) (hp : p ∈ E) (hf : f p ∈ arch (E.map f)) :
    ∀ q ∈ N p, dom (f q) (f p) = false := by
  intro q hq
  exact ((mem_arch _ _).mp hf).2 (f q) (List.mem_map.mpr ⟨q, hc p hp hf q hq, rfl⟩)

theorem fix_stable (f : α → V) (N : α → List α) (E : List α) (hc : Closed f N E) :
    proposals f N E ++ E = E := by
  rw [(closed_iff_no_proposals f N E).mpr hc]; rfl

omit [DecidableEq α] in
/-- The per-point form used when the archive keeps one representative per
    cell: only that representative's neighbours need to be evaluated. -/
theorem point_local_opt (f : α → V) (N : α → List α) (E : List α) (p : α)
    (hf : f p ∈ arch (E.map f)) (hn : ∀ q ∈ N p, q ∈ E) :
    ∀ q ∈ N p, dom (f q) (f p) = false := by
  intro q hq
  exact ((mem_arch _ _).mp hf).2 (f q) (List.mem_map.mpr ⟨q, hn q hq, rfl⟩)

/-! ### Resolution-bounded fronts

  At exact resolution the front can keep growing (it did: FIXPOINT run,
  doubling every round). At resolution `r` (objective `i` read in steps of
  `r_i`) the front is a front of cells. `coarse_le` says coarsening keeps
  the order; `coarse_front_within` says what a point on the coarse front
  is worth exactly: anything that dominates it does so by less than one
  resolution step in every objective. Halving `r` and continuing from the
  same evaluated set refines this bound without losing any evaluation. -/

/-- Read every objective in steps of `r_i` (floor). -/
def coarse : V → V → V
  | r :: rs, v :: vs => v / r :: coarse rs vs
  | _, _ => []

/-- `b` is below `a` by less than `r_i` in every objective. -/
def within : V → V → V → Prop
  | r :: rs, a :: as, b :: bs => a < b + r ∧ within rs as bs
  | _, _, _ => True

theorem div_mono (x y r : Nat) (h : x ≤ y) : x / r ≤ y / r := by
  rcases Nat.eq_zero_or_pos r with rfl | hr
  · simp
  · exact (Nat.le_div_iff_mul_le hr).mpr (Nat.le_trans (Nat.div_mul_le_self x r) h)

theorem coarse_le : ∀ (r a b : V), le a b = true → le (coarse r a) (coarse r b) = true
  | r :: rs, x :: xs, y :: ys, h => by
    simp only [le, Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [coarse, le, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨div_mono x y r h.1, coarse_le rs xs ys h.2⟩
  | [], a, b, h => by cases a <;> cases b <;> simp_all [coarse, le]
  | _ :: _, [], b, h => by cases b <;> simp_all [coarse, le]
  | _ :: _, _ :: _, [], h => by simp [le] at h

theorem le_not_dom_eq (a b : V) (h1 : le a b = true) (h2 : dom a b = false) : a = b := by
  simp only [dom, h1, Bool.true_and, bne_eq_false_iff_eq] at h2
  exact h2

theorem coarse_eq_within : ∀ (r a b : V), (∀ x ∈ r, 0 < x) → le b a = true →
    coarse r a = coarse r b → within r a b
  | [], _, _, _, _, _ => by simp [within]
  | r :: rs, [], _, _, _, _ => by simp [within]
  | r :: rs, x :: xs, [], _, _, _ => by simp [within]
  | r :: rs, x :: xs, y :: ys, hr, hl, he => by
    simp only [le, Bool.and_eq_true, decide_eq_true_eq] at hl
    simp only [coarse, List.cons.injEq] at he
    have hr0 : 0 < r := hr r List.mem_cons_self
    refine ⟨?_, coarse_eq_within rs xs ys (fun z hz => hr z (List.mem_cons_of_mem _ hz)) hl.2 he.2⟩
    have h1 := Nat.div_add_mod x r
    have h2 := Nat.div_add_mod y r
    have h3 := Nat.mod_lt x hr0
    have h4 := Nat.mod_lt y hr0
    rw [he.1] at h1
    have : x - y < r := by
      have hm : r * (y / r) ≤ y := Nat.mul_div_le y r
      omega
    omega

omit [DecidableEq α] in
/-- A point on the front of the coarsened objectives: whatever dominates it
    exactly does so by less than one resolution step in every objective. -/
theorem coarse_front_within (r : V) (hr : ∀ x ∈ r, 0 < x) (f : α → V) (E : List α) (p : α)
    (hf : coarse r (f p) ∈ arch (E.map (fun q => coarse r (f q)))) (q : α) (hq : q ∈ E)
    (hd : dom (f q) (f p) = true) : within r (f p) (f q) := by
  have hl : le (f q) (f p) = true := by simp only [dom, Bool.and_eq_true] at hd; exact hd.1
  have hc := coarse_le r _ _ hl
  have hnd := ((mem_arch _ _).mp hf).2 (coarse r (f q)) (List.mem_map.mpr ⟨q, hq, rfl⟩)
  exact coarse_eq_within r (f p) (f q) hr hl (le_not_dom_eq _ _ hc hnd).symm

end Prlm.Fix
