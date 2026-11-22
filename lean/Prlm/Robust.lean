import Prlm.Core
set_option warningAsError true

namespace Prlm.Robust

def keeps (earlier : Bool) (win rival : Int) : Prop :=
  if earlier then rival < win else rival ≤ win

def allowance (earlier : Bool) (win rival : Int) : Int :=
  win - rival - if earlier then 1 else 0

/-- The deterministic tie rule and the worst pair of signed disturbances
give an exact radius test against one rival. -/
theorem pair_iff (earlier : Bool) (win rival radius : Int) (hr : 0 ≤ radius) :
    (∀ dw dj : Int, -radius ≤ dw → dw ≤ radius →
      -radius ≤ dj → dj ≤ radius → keeps earlier (win + dw) (rival + dj)) ↔
    2 * radius ≤ allowance earlier win rival := by
  constructor
  · intro h
    have hcorner := h (-radius) radius (by omega) (by omega) (by omega) (by omega)
    cases earlier <;> simp [keeps, allowance] at hcorner ⊢ <;> omega
  · intro h dw dj hw₁ hw₂ hj₁ hj₂
    cases earlier <;> simp [keeps, allowance] at h ⊢ <;> omega

/-- A single disturbance lowers the chosen token and raises every rival,
so the conjunction of pair bounds is also exact for a whole vocabulary. -/
theorem vocabulary_iff {Token : Type} [DecidableEq Token]
    (scores : Token → Int) (chosen : Token) (earlier : Token → Bool)
    (radius : Int) (hr : 0 ≤ radius) :
    (∀ delta : Token → Int,
        (∀ t, -radius ≤ delta t ∧ delta t ≤ radius) →
        ∀ j, j ≠ chosen →
          keeps (earlier j) (scores chosen + delta chosen) (scores j + delta j)) ↔
    ∀ j, j ≠ chosen → 2 * radius ≤ allowance (earlier j) (scores chosen) (scores j) := by
  constructor
  · intro h j hj
    let corner : Token → Int := fun t => if t = chosen then -radius else radius
    have hbox : ∀ t, -radius ≤ corner t ∧ corner t ≤ radius := by
      intro t
      by_cases ht : t = chosen <;> simp [corner, ht] <;> omega
    have hcorner := h corner hbox j hj
    simp [corner, hj] at hcorner
    cases e : earlier j <;> simp [e, keeps, allowance] at hcorner ⊢ <;> omega
  · intro h delta hbox j hj
    exact (pair_iff (earlier j) (scores chosen) (scores j) radius hr).mpr
      (h j hj) (delta chosen) (delta j) (hbox chosen).1 (hbox chosen).2
      (hbox j).1 (hbox j).2

def least : List Int → Option Int
  | [] => none
  | hd :: tail => some (tail.foldl min hd)

theorem foldl_min_iff (threshold start : Int) (xs : List Int) :
    threshold ≤ xs.foldl min start ↔ threshold ≤ start ∧ ∀ x ∈ xs, threshold ≤ x := by
  induction xs generalizing start with
  | nil => simp
  | cons x xs ih =>
      simp only [List.foldl_cons, ih, Int.le_min, List.mem_cons]
      constructor
      · intro h
        exact ⟨h.1.1, fun y hy => hy.elim (fun e => e ▸ h.1.2) (h.2 y)⟩
      · intro h
        exact ⟨⟨h.1, h.2 x (Or.inl rfl)⟩, fun y hy => h.2 y (Or.inr hy)⟩

theorem least_iff (threshold : Int) (xs : List Int) :
    (match least xs with | none => True | some x => threshold ≤ x) ↔
    ∀ x ∈ xs, threshold ≤ x := by
  cases xs with
  | nil => simp [least]
  | cons a as => simp [least, foldl_min_iff]

def radius (chosen : Cand) (all : List Cand) : Option Int :=
  least ((all.filter (fun rival => rival.1 != chosen.1)).map
    (fun rival => allowance (decide (rival.1 < chosen.1)) chosen.2 rival.2 / 2))

theorem radius_iff (chosen : Cand) (all : List Cand) (r : Int) :
    (match radius chosen all with | none => True | some x => r ≤ x) ↔
    ∀ rival ∈ all, rival.1 ≠ chosen.1 →
      2 * r ≤ allowance (decide (rival.1 < chosen.1)) chosen.2 rival.2 := by
  rw [radius, least_iff]
  simp only [List.mem_map, List.mem_filter, bne_iff_ne]
  constructor
  · intro h rival haveMem neq
    have q := h _ ⟨rival, ⟨haveMem, neq⟩, rfl⟩
    omega
  · intro h d hd
    obtain ⟨rival, ⟨haveMem, neq⟩, rfl⟩ := hd
    have q := h rival haveMem neq
    omega

end Prlm.Robust
