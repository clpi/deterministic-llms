/-
  Exact sparse reweighting. A weight change moves only the candidates
  whose axis features are nonzero (the support S). The new pick is the
  better of (a) the moved candidates and (b) the best unmoved candidate,
  which a cached ordering of the old scores gives without rescanning.
  This replaces the margin box of `stable_of_bound` (a sufficient test
  that decided only 2% to 3% of real steps) with an exact rule whose
  work is |S| plus the rank of the first unmoved candidate.

  Proved: `best_split`: for any support S that contains every moved
  token, best (shift δ l) equals best over the moved candidates plus the
  best unmoved one.
-/
import Prlm.Core

namespace Prlm

theorem mem_toList {α : Type} {o : Option α} {a : α} : a ∈ o.toList ↔ o = some a := by
  cases o <;> simp [Option.toList, eq_comm]

theorem best_split (l : List Cand) (S : Nat → Bool) (δ : Nat → Int)
    (hz : ∀ c ∈ l, S c.1 = false → δ c.1 = 0) :
    best (shift δ l) =
      best (shift δ (l.filter (fun c => S c.1)) ++ (best (l.filter (fun c => !S c.1))).toList) := by
  -- every candidate on the right is a candidate on the left
  have sub : ∀ y, y ∈ shift δ (l.filter (fun c => S c.1)) ++ (best (l.filter (fun c => !S c.1))).toList →
      y ∈ shift δ l := by
    intro y hy
    rcases List.mem_append.mp hy with h | h
    · obtain ⟨x, hx, rfl⟩ := mem_shift.mp h
      exact mem_shift.mpr ⟨x, (List.mem_filter.mp hx).1, rfl⟩
    · have hb := best_mem (mem_toList.mp h)
      have hxl := (List.mem_filter.mp hb).1
      have hS : S y.1 = false := by simpa using (List.mem_filter.mp hb).2
      exact mem_shift.mpr ⟨y, hxl, by rw [hz y hxl hS]; simp⟩
  -- the right side's best is unbeaten on the left
  have top : ∀ r', best (shift δ (l.filter (fun c => S c.1)) ++ (best (l.filter (fun c => !S c.1))).toList) = some r' →
      ∀ y ∈ shift δ l, ¬ Better y r' := by
    intro r' hr' y hy
    obtain ⟨x, hx, rfl⟩ := mem_shift.mp hy
    cases hs : S x.1
    · -- unmoved: compare with the best unmoved candidate
      have hxf : x ∈ l.filter (fun c => !S c.1) := List.mem_filter.mpr ⟨hx, by simp [hs]⟩
      cases hb : best (l.filter (fun c => !S c.1)) with
      | none => rw [best_none] at hb; rw [hb] at hxf; simp at hxf
      | some b =>
        have h1 : ¬ Better x b := best_max hb x hxf
        have h2 : ¬ Better b r' := best_max hr' b (List.mem_append_right _ (by simp [hb]))
        have : (x.1, x.2 + δ x.1) = x := by rw [hz x hx hs]; simp
        rw [this]; exact not_better_trans h1 h2
    · have hm : (x.1, x.2 + δ x.1) ∈ shift δ (l.filter (fun c => S c.1)) :=
        mem_shift.mpr ⟨x, List.mem_filter.mpr ⟨hx, by simp [hs]⟩, rfl⟩
      exact best_max hr' _ (List.mem_append_left _ hm)
  cases hl : best (shift δ l) with
  | none =>
    rw [best_none] at hl
    symm; rw [best_none]
    cases e : shift δ (l.filter (fun c => S c.1)) ++ (best (l.filter (fun c => !S c.1))).toList with
    | nil => rfl
    | cons y ys =>
      have := sub y (by rw [e]; exact List.mem_cons_self)
      rw [hl] at this; simp at this
  | some r =>
    cases hr : best (shift δ (l.filter (fun c => S c.1)) ++ (best (l.filter (fun c => !S c.1))).toList) with
    | none =>
      exfalso
      rw [best_none] at hr
      obtain ⟨x, hx, rfl⟩ := mem_shift.mp (best_mem hl)
      cases hs : S x.1
      · cases hb : best (l.filter (fun c => !S c.1)) with
        | none =>
          rw [best_none] at hb
          have : x ∈ l.filter (fun c => !S c.1) := List.mem_filter.mpr ⟨hx, by simp [hs]⟩
          rw [hb] at this; simp at this
        | some b => rw [hb] at hr; simp at hr
      · have : (x.1, x.2 + δ x.1) ∈ shift δ (l.filter (fun c => S c.1)) :=
          mem_shift.mpr ⟨x, List.mem_filter.mpr ⟨hx, by simp [hs]⟩, rfl⟩
        rw [List.append_eq_nil_iff] at hr; rw [hr.1] at this; simp at this
    | some r' =>
      have h1 : ¬ Better r' r := best_max hl r' (sub r' (best_mem hr))
      have h2 : ¬ Better r r' := top r' hr r (best_mem hl)
      rw [max_unique h2 h1]

end Prlm
