/-
  The self-improving search loop's soundness (tasks/LOOP_CONTRACT.json).

  The loop appends candidate weight changes to the one archive
  (AlgoSearch.arch), admits or rejects each by checking the archive alone,
  drops axes that cannot move any decision, and refines around the front.
  These theorems are what make that loop safe to run without supervision:

  * arch_cons_sub   one append changes the front only by the new point
  * arch_covers     appending any batch never loses ground: every old front
                    point stays on the front or is dominated by a new one
  * admit_sound     no archived point dominates x  =>  x is on the new front
  * reject_sound    an archived point dominates x  =>  x is not
  * scope_sound     an axis that is zero on every candidate of every step
                    changes no decision, so no objective, at any weight
-/
import Prlm.Core
import Prlm.AlgoSearch

namespace Prlm.Loop
open AlgoSearch

theorem arch_cons_sub (x : V) (L : List V) (z : V) (h : z ∈ arch (x :: L)) :
    z = x ∨ z ∈ arch L := by
  rw [mem_arch] at h
  obtain ⟨hm, hnd⟩ := h
  rcases List.mem_cons.mp hm with e | hl
  · exact Or.inl e
  · right
    rw [mem_arch]
    exact ⟨hl, fun y hy => hnd y (List.mem_cons.mpr (Or.inr hy))⟩

theorem arch_covers (L M : List V) (y : V) (hy : y ∈ arch L) :
    y ∈ arch (M ++ L) ∨ ∃ m, m ∈ arch (M ++ L) ∧ dom m y = true := by
  have hyL : y ∈ L := ((mem_arch L y).mp hy).1
  have hyML : y ∈ M ++ L := List.mem_append.mpr (Or.inr hyL)
  by_cases h : ∃ z, z ∈ M ++ L ∧ dom z y = true
  · obtain ⟨z, hz, hd⟩ := h
    obtain ⟨m, hm, hmd⟩ := front_dominator (M ++ L) y z.sum z rfl hz hd
    exact Or.inr ⟨m, (mem_arch _ _).mpr hm, hmd⟩
  · left
    rw [mem_arch]
    refine ⟨hyML, fun z hz => ?_⟩
    cases e : dom z y with
    | false => rfl
    | true => exact absurd ⟨z, hz, e⟩ h

theorem admit_sound (L : List V) (x : V) (h : ∀ m, m ∈ arch L → dom m x = false) :
    x ∈ arch (x :: L) := by
  rw [mem_arch]
  refine ⟨List.mem_cons.mpr (Or.inl rfl), fun y hy => ?_⟩
  rcases List.mem_cons.mp hy with e | hl
  · rw [e]; exact dom_irrefl _
  · cases e : dom y x with
    | false => rfl
    | true =>
      obtain ⟨m, hm, hmd⟩ := front_dominator L x y.sum y rfl hl e
      have h2 := h m ((mem_arch L m).mpr hm)
      rw [h2] at hmd
      exact absurd hmd (by decide)

theorem reject_sound (L : List V) (x m : V) (hm : m ∈ arch L) (hd : dom m x = true) :
    x ∉ arch (x :: L) := by
  rw [mem_arch]
  intro hx
  have h2 := hx.2 m (List.mem_cons.mpr (Or.inr ((mem_arch L m).mp hm).1))
  rw [h2] at hd
  exact absurd hd (by decide)

/-- One recorded step's decision under weights θ. -/
def decide1 (s : List Cand × (Nat → List Int)) (θ : List Int) : Option Nat :=
  (best (scores s.1 s.2 θ)).map Prod.fst

theorem scope_sound (steps : List (List Cand × (Nat → List Int))) (θ : List Int) (a : Nat) (x : Int)
    (hz : ∀ s ∈ steps, ∀ c ∈ s.1, (s.2 c.1).getD a 0 = 0)
    {β : Type} (obj : List (Option Nat) → β) :
    obj (steps.map (fun s => decide1 s (setAt θ a x))) = obj (steps.map (fun s => decide1 s θ)) := by
  congr 1
  apply List.map_congr_left
  intro s hs
  simp only [decide1]
  rw [scores_local s.1 s.2 θ a x (hz s hs)]

end Prlm.Loop
