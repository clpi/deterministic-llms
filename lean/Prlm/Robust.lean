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

namespace Prlm.Robust

theorem interval_box_iff {Token : Type} [DecidableEq Token]
    (scores lower upper : Token → Int) (chosen : Token) (earlier : Token → Bool)
    (valid : ∀ t, lower t ≤ upper t) :
    (∀ delta : Token → Int,
        (∀ t, lower t ≤ delta t ∧ delta t ≤ upper t) →
        ∀ j, j ≠ chosen →
          keeps (earlier j) (scores chosen + delta chosen) (scores j + delta j)) ↔
    ∀ j, j ≠ chosen →
      upper j - lower chosen ≤ allowance (earlier j) (scores chosen) (scores j) := by
  constructor
  · intro h j hj
    let corner : Token → Int := fun t => if t = chosen then lower t else upper t
    have hbox : ∀ t, lower t ≤ corner t ∧ corner t ≤ upper t := by
      intro t
      by_cases ht : t = chosen <;> simp [corner, ht, valid]
    have hw := h corner hbox j hj
    simp [corner, hj] at hw
    cases e : earlier j <;> simp [e, keeps, allowance] at hw ⊢ <;> omega
  · intro h delta hbox j hj
    have hgap := h j hj
    have hjhi := (hbox j).2
    have hwlo := (hbox chosen).1
    cases e : earlier j <;> simp [e, keeps, allowance] at hgap ⊢ <;> omega

theorem relative_iff (earlier : Bool) (win rival dw dj : Int) :
    keeps earlier (win + dw) (rival + dj) ↔
      dj - dw ≤ allowance earlier win rival := by
  cases earlier <;> simp [keeps, allowance] <;> omega

theorem uniform_drift_preserves (earlier : Bool) (win rival d : Int) :
    keeps earlier (win + d) (rival + d) ↔ keeps earlier win rival := by
  cases earlier <;> simp [keeps] <;> omega

def trace {Token : Type} (step : List Token → Token) : Nat → List Token → List Token
  | 0, pre => pre
  | n + 1, pre => trace step n (pre ++ [step pre])

theorem trace_eq_of_on_reference_prefix {Token : Type}
    (ref candidate : List Token → Token) (n : Nat) (pre : List Token)
    (agree : ∀ k < n, candidate (trace ref k pre) = ref (trace ref k pre)) :
    trace candidate n pre = trace ref n pre := by
  induction n generalizing pre with
  | zero => rfl
  | succ n ih =>
    have first := agree 0 (by omega)
    change candidate pre = ref pre at first
    simp only [trace, first]
    apply ih
    intro k hk
    exact agree (k + 1) (by omega)

end Prlm.Robust

namespace Prlm.Robust

def checkedOrFallback {Token : Type} (checked : List Token → Option Token)
    (cold : List Token → Token) (pre : List Token) : Token :=
  match checked pre with
  | some t => t
  | none => cold pre

/-- A miss uses the exact cold step. It does not invalidate later hits on
the same reference trajectory. Evidence is needed only at prefixes the
reference actually reaches, and the selected value may carry a receipt. -/
theorem checked_or_fallback_trace {Token : Type}
    (checked : List Token → Option Token) (cold : List Token → Token)
    (n : Nat) (pre : List Token)
    (sound : ∀ k < n, ∀ t, checked (trace cold k pre) = some t →
      t = cold (trace cold k pre)) :
    trace (checkedOrFallback checked cold) n pre = trace cold n pre := by
  apply trace_eq_of_on_reference_prefix
  intro k hk
  simp only [checkedOrFallback]
  split
  · rename_i token h
    exact sound k hk token h
  · rfl

end Prlm.Robust

namespace Prlm.Robust

def projectLast {Hidden Score : Type} (head : Hidden → Score) (rows : List Hidden) :
    Option Score :=
  match rows.getLast? with
  | none => none
  | some row => some (head row)

/-- The requested head output can be formed directly from its input row.
This is a row-local mathematical model, not a claim about floating kernels. -/
theorem head_project_last {Hidden Score : Type} (head : Hidden → Score) (rows : List Hidden) :
    (rows.map head).getLast? = projectLast head rows := by
  simp only [List.getLast?_map, projectLast]
  cases rows.getLast? <;> rfl

end Prlm.Robust
