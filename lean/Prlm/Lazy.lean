import Prlm.Portfolio
/-!
The lazy ratchet (tasks/VALUE_AMEND_6.json, claims L1 and L2).

L1: producing candidates one at a time and stopping at the first that passes the
checker selects exactly what `firstPass` over the whole list selects, after
evaluating at most the whole list; the evaluated prefix alone decides it.

L2: once one answer in the produced prefix leads every other answer by more
than the number of candidates still to come, that answer has strictly the
highest count in the whole list, so every majority rule (whatever its tie
rule) returns it. Stopping there cannot change the vote.

Accuracy and tokens are measured, not proved.
-/
namespace Prlm.Lazy
open Prlm.Portfolio

/-- Evaluate in order, stop at the first candidate passing `c`; returns the
selection and the number of candidates evaluated. -/
def lazyFirst {α : Type} (c : α → Bool) : List α → Option α × Nat
  | [] => (none, 0)
  | x :: xs => if c x then (some x, 1) else ((lazyFirst c xs).1, (lazyFirst c xs).2 + 1)

/-- L1: the lazy selection is the eager one. -/
theorem lazyFirst_sel {α : Type} (c : α → Bool) :
    ∀ L : List α, (lazyFirst c L).1 = firstPass c L
  | [] => rfl
  | x :: xs => by
    by_cases hx : c x = true
    · simp [lazyFirst, firstPass, hx]
    · simp [lazyFirst, firstPass, hx, lazyFirst_sel c xs]

/-- L1: never more evaluations than candidates. -/
theorem lazyFirst_cost_le {α : Type} (c : α → Bool) :
    ∀ L : List α, (lazyFirst c L).2 ≤ L.length
  | [] => by simp [lazyFirst]
  | x :: xs => by
    by_cases hx : c x = true
    · simp [lazyFirst, hx]
    · have := lazyFirst_cost_le c xs
      simp [lazyFirst, hx]
      omega

/-- L1: the evaluated prefix alone decides the selection. -/
theorem lazyFirst_prefix {α : Type} (c : α → Bool) :
    ∀ L : List α, firstPass c (L.take (lazyFirst c L).2) = firstPass c L
  | [] => rfl
  | x :: xs => by
    by_cases hx : c x = true
    · simp [lazyFirst, firstPass, hx]
    · simp [lazyFirst, firstPass, hx, lazyFirst_prefix c xs]

/-- Candidates after a passing prefix never matter. -/
theorem firstPass_append_some {α : Type} (c : α → Bool) :
    ∀ (l1 l2 : List α) (y : α), firstPass c l1 = some y → firstPass c (l1 ++ l2) = some y
  | [], _, _, h => by simp [firstPass] at h
  | x :: xs, l2, y, h => by
    by_cases hx : c x = true
    · simp [firstPass, hx] at h ⊢; exact h
    · simp [firstPass, hx] at h ⊢; exact firstPass_append_some c xs l2 y h

/-- After a prefix with no passing candidate, only the rest matters. -/
theorem firstPass_append_none {α : Type} (c : α → Bool) :
    ∀ (l1 l2 : List α), firstPass c l1 = none → firstPass c (l1 ++ l2) = firstPass c l2
  | [], _, _ => rfl
  | x :: xs, l2, h => by
    by_cases hx : c x = true
    · simp [firstPass, hx] at h
    · simp [firstPass, hx] at h ⊢; exact firstPass_append_none c xs l2 h

/-- L2: a lead larger than the number of candidates left survives them. -/
theorem decided {α : Type} [DecidableEq α] (p rest : List α) (a : α)
    (h : ∀ b, b ≠ a → p.count b + rest.length < p.count a) :
    ∀ b, b ≠ a → (p ++ rest).count b < (p ++ rest).count a := by
  intro b hb
  have hb' := h b hb
  have hr : rest.count b ≤ rest.length := List.count_le_length
  simp only [List.count_append]
  omega

/-- L2: every answer with the highest count in the whole list is the leader. -/
theorem decided_mode {α : Type} [DecidableEq α] (p rest : List α) (a : α)
    (h : ∀ b, b ≠ a → p.count b + rest.length < p.count a) (m : α)
    (hm : ∀ b, (p ++ rest).count b ≤ (p ++ rest).count m) : m = a := by
  by_cases hma : m = a
  · exact hma
  · have := decided p rest a h m hma
    have := hm a
    omega

end Prlm.Lazy
