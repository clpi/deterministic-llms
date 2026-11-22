/-
  Output structure as a parametric, certified, reversible decision
  (tasks/STRUCT_CONTRACT.json).

  A reply's structure (essay, song, steps, demo, table, ...) is one more
  certified pick among candidates with integer scores, so everything
  proved in Core applies. This file adds what the structure layer needs:

  * picked_iff        a bias shift Δ on structure p makes p the pick exactly
                      when p beats every rival outright (`Wins`)
  * minshift_wins     the shift `dstar` makes the revealed structure win
  * minshift_minimal  no smaller non-negative shift does: the update is
                      neither under-tuned nor over-tuned for that event
  * minshiftAll_*     the same for a set of events at once (the maximum)
  * rescore_comm      weight updates compose in any order
  * replan_local      replacing one segment's inputs changes only that
                      segment's decision
  * parse_render      rendering is reversible: parse recovers structure and
                      segments exactly
  * render_set        re-rendering after one segment changes rewrites only
                      that segment's slice of the output
-/
import Prlm.Core
import Prlm.Loop

namespace Prlm.Struct
open Prlm

/-! ### Minimal shift for a revealed preference -/

/-- Shift structure `p` (current score `sp`) needs to beat rival `y`
    outright; ties go to the lower id, as in `Better`. -/
def need (p : Nat) (sp : Int) (y : Cand) : Int :=
  y.2 - sp + (if y.1 < p then 1 else 0)

def Wins (p : Nat) (sp : Int) (rivals : List Cand) (Δ : Int) : Prop :=
  ∀ y ∈ rivals, Better (p, sp + Δ) y

theorem wins_iff (p : Nat) (sp : Int) (rivals : List Cand) (Δ : Int)
    (hr : ∀ y ∈ rivals, y.1 ≠ p) :
    Wins p sp rivals Δ ↔ ∀ y ∈ rivals, need p sp y ≤ Δ := by
  unfold Wins need Better
  constructor
  · intro h y hy
    have h1 := h y hy; have h2 := hr y hy
    dsimp only at h1
    split <;> omega
  · intro h y hy
    have h1 := h y hy; have h2 := hr y hy
    dsimp only
    split at h1 <;> omega

def dstar (p : Nat) (sp : Int) : List Cand → Int
  | [] => 0
  | y :: ys => max (need p sp y) (dstar p sp ys)

theorem dstar_nonneg (p : Nat) (sp : Int) : ∀ l : List Cand, 0 ≤ dstar p sp l
  | [] => by simp [dstar]
  | y :: ys => by have := dstar_nonneg p sp ys; simp only [dstar]; omega

theorem need_le_dstar (p : Nat) (sp : Int) : ∀ (l : List Cand), ∀ y ∈ l, need p sp y ≤ dstar p sp l
  | [], y, hy => by simp at hy
  | x :: xs, y, hy => by
    simp only [dstar]
    rcases List.mem_cons.mp hy with rfl | hy
    · omega
    · have := need_le_dstar p sp xs y hy; omega

theorem dstar_attained (p : Nat) (sp : Int) :
    ∀ l : List Cand, dstar p sp l = 0 ∨ ∃ y ∈ l, need p sp y = dstar p sp l
  | [] => by simp [dstar]
  | x :: xs => by
    simp only [dstar]
    rcases dstar_attained p sp xs with h | ⟨y, hy, he⟩
    · by_cases hx : need p sp x ≤ 0
      · left; omega
      · right; exact ⟨x, List.mem_cons_self, by omega⟩
    · by_cases hx : dstar p sp xs ≤ need p sp x
      · right; exact ⟨x, List.mem_cons_self, by omega⟩
      · right; exact ⟨y, List.mem_cons_of_mem _ hy, by omega⟩

theorem minshift_wins (p : Nat) (sp : Int) (rivals : List Cand) (hr : ∀ y ∈ rivals, y.1 ≠ p) :
    Wins p sp rivals (dstar p sp rivals) :=
  (wins_iff p sp rivals _ hr).mpr (need_le_dstar p sp rivals)

theorem minshift_minimal (p : Nat) (sp : Int) (rivals : List Cand) (hr : ∀ y ∈ rivals, y.1 ≠ p)
    (Δ : Int) (h0 : 0 ≤ Δ) (hlt : Δ < dstar p sp rivals) : ¬ Wins p sp rivals Δ := by
  rw [wins_iff p sp rivals Δ hr]
  intro h
  rcases dstar_attained p sp rivals with he | ⟨y, hy, he⟩
  · omega
  · have := h y hy; omega

/-- The shift on structure `p` alone. -/
def bias (p : Nat) (Δ : Int) : Nat → Int := fun t => if t = p then Δ else 0

theorem shift_rivals (p : Nat) (Δ : Int) (rivals : List Cand) (hr : ∀ y ∈ rivals, y.1 ≠ p) :
    shift (bias p Δ) rivals = rivals := by
  unfold shift
  conv => rhs; rw [← List.map_id rivals]
  apply List.map_congr_left
  intro y hy
  simp [bias, hr y hy]

/-- With the bias shift, `p` is picked exactly when it wins outright. -/
theorem picked_iff (p : Nat) (sp : Int) (rivals : List Cand) (hr : ∀ y ∈ rivals, y.1 ≠ p)
    (Δ : Int) :
    (best (shift (bias p Δ) ((p, sp) :: rivals))).map Prod.fst = some p ↔ Wins p sp rivals Δ := by
  have hs : shift (bias p Δ) ((p, sp) :: rivals) = (p, sp + Δ) :: rivals := by
    show (p, sp + bias p Δ p) :: shift (bias p Δ) rivals = _
    rw [shift_rivals p Δ rivals hr]; simp [bias]
  rw [hs]
  cases hb : best ((p, sp + Δ) :: rivals) with
  | none => simp [best_none] at hb
  | some r =>
    simp only [Option.map_some', Option.some.injEq]
    have hm := best_mem hb
    have hx := best_max hb
    constructor
    · intro hrp y hy
      rcases List.mem_cons.mp hm with rfl | hrr
      · by_cases hb' : Better (p, sp + Δ) y
        · exact hb'
        · exfalso
          have hny := hx y (List.mem_cons_of_mem _ hy)
          have := max_unique hny hb'
          exact hr y hy (by rw [← this])
      · exact absurd hrp (hr r hrr)
    · intro hw
      rcases List.mem_cons.mp hm with rfl | hrr
      · rfl
      · exact absurd (hw r hrr) (hx _ List.mem_cons_self)

/-- Several revealed-preference events for the same structure `p`
    (each a current score and its rivals): the shift that serves all. -/
def dstarAll (p : Nat) : List (Int × List Cand) → Int
  | [] => 0
  | e :: es => max (dstar p e.1 e.2) (dstarAll p es)

theorem minshiftAll_wins (p : Nat) : ∀ (es : List (Int × List Cand)),
    (∀ e ∈ es, ∀ y ∈ e.2, y.1 ≠ p) → ∀ e ∈ es, Wins p e.1 e.2 (dstarAll p es)
  | [], _, e, he => by simp at he
  | x :: xs, hr, e, he => by
    rw [wins_iff p e.1 e.2 _ (hr e he)]
    intro y hy
    simp only [dstarAll]
    rcases List.mem_cons.mp he with rfl | he'
    · have := need_le_dstar p e.1 e.2 y hy; omega
    · have h := (wins_iff p e.1 e.2 _ (hr e (List.mem_cons_of_mem _ he'))).mp
        (minshiftAll_wins p xs (fun e h => hr e (List.mem_cons_of_mem _ h)) e he') y hy
      omega

theorem minshiftAll_minimal (p : Nat) : ∀ (es : List (Int × List Cand)),
    (∀ e ∈ es, ∀ y ∈ e.2, y.1 ≠ p) → ∀ Δ, 0 ≤ Δ → Δ < dstarAll p es →
    ∃ e ∈ es, ¬ Wins p e.1 e.2 Δ
  | [], _, Δ, _, h => by simp [dstarAll] at h; omega
  | x :: xs, hr, Δ, h0, h => by
    simp only [dstarAll] at h
    by_cases hx : Δ < dstar p x.1 x.2
    · exact ⟨x, List.mem_cons_self, minshift_minimal p x.1 x.2 (hr x List.mem_cons_self) Δ h0 hx⟩
    · obtain ⟨e, he, hne⟩ := minshiftAll_minimal p xs (fun e h => hr e (List.mem_cons_of_mem _ h))
        Δ h0 (by omega)
      exact ⟨e, List.mem_cons_of_mem _ he, hne⟩

/-! ### Composition and incremental replanning -/

theorem rescore_comm (Δ1 Δ2 : List Int) (feat : Nat → List Int) (l : List Cand) :
    rescore Δ1 feat (rescore Δ2 feat l) = rescore Δ2 feat (rescore Δ1 feat l) := by
  simp only [rescore, shift, List.map_map]
  apply List.map_congr_left
  intro c _
  simp only [Function.comp, Prod.mk.injEq, true_and]
  omega

theorem replan_local (steps : List (List Cand × (Nat → List Int))) (θ : List Int) (i : Nat)
    (s' : List Cand × (Nat → List Int)) :
    (steps.set i s').map (fun s => Loop.decide1 s θ) =
      (steps.map (fun s => Loop.decide1 s θ)).set i (Loop.decide1 s' θ) := by
  rw [List.map_set]

/-! ### Reversible rendering -/

/-- Body: each segment followed by the separator token. -/
def body (SEP : Nat) (segs : List (List Nat)) : List Nat := segs.flatMap (fun s => s ++ [SEP])

def render (SEP f : Nat) (segs : List (List Nat)) : List Nat := f :: body SEP segs

def splitSep (SEP : Nat) : List Nat → List Nat → List (List Nat)
  | _, [] => []
  | acc, t :: ts => if t = SEP then acc.reverse :: splitSep SEP [] ts else splitSep SEP (t :: acc) ts

def parse (SEP : Nat) : List Nat → Option (Nat × List (List Nat))
  | [] => none
  | f :: b => some (f, splitSep SEP [] b)

theorem splitSep_seg (SEP : Nat) : ∀ (s acc rest : List Nat), SEP ∉ s →
    splitSep SEP acc (s ++ SEP :: rest) = (acc.reverse ++ s) :: splitSep SEP [] rest
  | [], acc, rest, _ => by simp [splitSep]
  | t :: ts, acc, rest, h => by
    have ht : t ≠ SEP := fun e => h (e ▸ List.mem_cons_self)
    have hts : SEP ∉ ts := fun e => h (List.mem_cons_of_mem _ e)
    simp only [List.cons_append, splitSep, if_neg ht]
    rw [splitSep_seg SEP ts (t :: acc) rest hts]
    simp

theorem split_body (SEP : Nat) : ∀ segs : List (List Nat), (∀ s ∈ segs, SEP ∉ s) →
    splitSep SEP [] (body SEP segs) = segs
  | [], _ => by simp [body, splitSep]
  | s :: ss, h => by
    have h1 : body SEP (s :: ss) = s ++ SEP :: body SEP ss := by simp [body]
    rw [h1, splitSep_seg SEP s [] _ (h s List.mem_cons_self),
      split_body SEP ss (fun x hx => h x (List.mem_cons_of_mem _ hx))]
    simp

theorem parse_render (SEP f : Nat) (segs : List (List Nat)) (h : ∀ s ∈ segs, SEP ∉ s) :
    parse SEP (render SEP f segs) = some (f, segs) := by
  simp [parse, render, split_body SEP segs h]

theorem render_set (SEP f : Nat) (segs : List (List Nat)) (i : Nat) (s' : List Nat)
    (hi : i < segs.length) :
    render SEP f (segs.set i s') =
      f :: (body SEP (segs.take i) ++ (s' ++ [SEP]) ++ body SEP (segs.drop (i + 1))) := by
  simp only [render, body]
  rw [List.set_eq_take_append_cons_drop, if_pos hi]
  simp

end Prlm.Struct
