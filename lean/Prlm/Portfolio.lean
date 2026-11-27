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

end Prlm.Portfolio
