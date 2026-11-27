/-
Portfolio dominance (tasks/VALUE_AMEND_3.json, claim P1).

A decoder that keeps every candidate of several baselines and returns the
first candidate passing a checker passes the checker whenever any baseline's
candidate does. Nothing here is about hidden tests or compute: those are
measured, not proved.
-/
namespace Prlm.Portfolio

/-- The first element of `U` that passes `c`. -/
def firstPass {α : Type} (c : α → Bool) : List α → Option α
  | [] => none
  | x :: xs => if c x then some x else firstPass c xs

/-- Whatever `firstPass` returns passes the checker and is in the list. -/
theorem firstPass_spec {α : Type} (c : α → Bool) :
    ∀ (U : List α) (y : α), firstPass c U = some y → c y = true ∧ y ∈ U
  | [], y, h => by simp [firstPass] at h
  | x :: xs, y, h => by
    unfold firstPass at h
    by_cases hx : c x = true
    · simp [hx] at h
      subst h
      exact ⟨hx, List.mem_cons_self⟩
    · simp [hx] at h
      have ⟨h1, h2⟩ := firstPass_spec c xs y h
      exact ⟨h1, List.mem_cons_of_mem x h2⟩

/-- If some element of `U` passes `c`, `firstPass` finds one. -/
theorem firstPass_some {α : Type} (c : α → Bool) :
    ∀ (U : List α) (x : α), x ∈ U → c x = true → ∃ y, firstPass c U = some y
  | [], x, hx, _ => by simp at hx
  | z :: zs, x, hx, hc => by
    unfold firstPass
    by_cases hz : c z = true
    · exact ⟨z, by simp [hz]⟩
    · simp [hz]
      rcases List.mem_cons.mp hx with h | h
      · subst h; exact absurd hc hz
      · exact firstPass_some c zs x h hc

/-- P1: a portfolio containing every candidate of `A` passes the checker
whenever some candidate of `A` does. -/
theorem portfolio_dominates {α : Type} (c : α → Bool) (A U : List α)
    (hsub : ∀ x, x ∈ A → x ∈ U) (hA : ∃ x, x ∈ A ∧ c x = true) :
    ∃ y, firstPass c U = some y ∧ c y = true := by
  obtain ⟨x, hxA, hcx⟩ := hA
  obtain ⟨y, hy⟩ := firstPass_some c U x (hsub x hxA) hcx
  exact ⟨y, hy, (firstPass_spec c U y hy).1⟩

/-- The same for several baselines at once: each baseline's selection is
some candidate of its own list; if any of them passes, so does the portfolio. -/
theorem portfolio_dominates_all {α : Type} (c : α → Bool) (Bs : List (List α)) (U : List α)
    (hsub : ∀ B, B ∈ Bs → ∀ x, x ∈ B → x ∈ U)
    (h : ∃ B, B ∈ Bs ∧ ∃ x, x ∈ B ∧ c x = true) :
    ∃ y, firstPass c U = some y ∧ c y = true := by
  obtain ⟨B, hB, hx⟩ := h
  exact portfolio_dominates c B U (hsub B hB) hx

/-- Q1 (VALUE_AMEND_4): along nested candidate lists, once a step's selection
passes the checker, every later step's selection passes too. `steps n` is the
candidate list of step n; `hnest` says each step keeps the previous one. -/
theorem ratchet_monotone {α : Type} (c : α → Bool) (steps : Nat → List α)
    (hnest : ∀ n x, x ∈ steps n → x ∈ steps (n + 1)) :
    ∀ n k y, firstPass c (steps n) = some y → c y = true →
      ∃ z, firstPass c (steps (n + k)) = some z ∧ c z = true := by
  intro n k
  induction k with
  | zero => intro y hy hc; exact ⟨y, by simpa using hy, hc⟩
  | succ k ih =>
    intro y hy hc
    obtain ⟨z, hz, hcz⟩ := ih y hy hc
    have hzmem := (firstPass_spec c (steps (n + k)) z hz).2
    exact portfolio_dominates c (steps (n + k)) (steps (n + k + 1))
      (hnest (n + k)) ⟨z, hzmem, hcz⟩

/-- Candidates of `L` that a cache holding `prev` must still evaluate. -/
def fresh {α : Type} [DecidableEq α] (prev L : List α) : List α :=
  L.filter (fun x => !(prev.contains x))

/-- Q2 (VALUE_AMEND_4): a step's new evaluations are exactly its candidates
not in the previous step, and never more than the step's own size. -/
theorem fresh_spec {α : Type} [DecidableEq α] (prev L : List α) (x : α) :
    x ∈ fresh prev L ↔ x ∈ L ∧ x ∉ prev := by
  simp [fresh, List.mem_filter]

theorem fresh_le {α : Type} [DecidableEq α] (prev L : List α) :
    (fresh prev L).length ≤ L.length := by
  unfold fresh
  exact List.length_filter_le _ _

/-- Re-evaluating the previous step costs nothing. -/
theorem fresh_self {α : Type} [DecidableEq α] (L : List α) : fresh L L = [] := by
  unfold fresh
  rw [List.filter_eq_nil_iff]
  intro x hx
  simp [List.contains_iff_mem, hx]

end Prlm.Portfolio
