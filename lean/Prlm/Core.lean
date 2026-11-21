/-
  PRLM core: the certified best-first rule, named-axis scoring,
  reweighting, and generation. These are the definitions the
  executable `prlm` runs on real model scores (Main.lean); every
  theorem here is about those same functions.

  A candidate is an edge from the current content to a token, carrying
  an integer score. The base score is the model's (any function of the
  content); each named axis adds a weight times the axis's feature on
  that edge (the axis is a coloring of the edge). Integer arithmetic
  throughout, so every statement is exact.

  Proved:
  * `pickF_mem`, `pickF_max`, `max_unique`: the pick is a candidate,
    nothing beats it, and the unbeaten candidate is unique.
  * `best_perm`: the pick does not depend on candidate order
    (determinism under any enumeration or merge order).
  * `certify_sound`: a certified pick strictly beats every other
    token by at least the margin (nothing left can beat or tie it).
  * `certify_complete`: if some candidate strictly beats every other
    token, the rule picks it and certifies it.
  * `rescore_eq`: reweighting the scored list by a weight change
    equals recomputing the scores with the new weights.
  * `stable_of_bound`: inside the cone given by the margin
    (every score moves by less than half the margin), the pick does
    not change, so no recompute is needed to know it.
  * `scores_local`, `dot_set_zero`: changing the weight of an axis
    whose feature is zero on every candidate changes nothing.
  * generation: `gen_prefix`, `gen_perm`, `gen_local`, `gen_cache`,
    `gen_reweight`: outputs extend their prompts, do not depend on
    candidate order, ignore axes that are zero on every content,
    are unchanged by a correct cache, and the reweighted run that
    reuses cached base scores equals the run from scratch.
-/

namespace Prlm

/-- A candidate edge: token and integer score. -/
abbrev Cand := Nat × Int

/-- Strict order: higher score first, then lower token id. -/
def Better (a b : Cand) : Prop := b.2 < a.2 ∨ (a.2 = b.2 ∧ a.1 < b.1)

instance (a b : Cand) : Decidable (Better a b) := by unfold Better; infer_instance

theorem better_trans {a b c : Cand} (h1 : Better a b) (h2 : Better b c) : Better a c := by
  obtain ⟨a1, a2⟩ := a; obtain ⟨b1, b2⟩ := b; obtain ⟨c1, c2⟩ := c
  simp only [Better] at *; omega

theorem not_better_trans {a b c : Cand} (h1 : ¬ Better a b) (h2 : ¬ Better b c) :
    ¬ Better a c := by
  obtain ⟨a1, a2⟩ := a; obtain ⟨b1, b2⟩ := b; obtain ⟨c1, c2⟩ := c
  simp only [Better] at *; omega

theorem better_irrefl (a : Cand) : ¬ Better a a := by
  obtain ⟨a1, a2⟩ := a; simp only [Better]; omega

/-- One left-to-right pass keeping the better candidate. -/
def pickF (m : Cand) : List Cand → Cand
  | [] => m
  | x :: xs => pickF (if Better x m then x else m) xs

def best : List Cand → Option Cand
  | [] => none
  | c :: cs => some (pickF c cs)

theorem pickF_mem : ∀ (m : Cand) (l : List Cand), pickF m l ∈ m :: l
  | m, [] => by simp [pickF]
  | m, x :: xs => by
    have ih := pickF_mem (if Better x m then x else m) xs
    simp only [pickF]
    by_cases h : Better x m
    · rw [if_pos h] at ih ⊢
      rcases List.mem_cons.mp ih with e | e
      · rw [e]; exact List.mem_cons_of_mem _ List.mem_cons_self
      · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ e)
    · rw [if_neg h] at ih ⊢
      rcases List.mem_cons.mp ih with e | e
      · rw [e]; exact List.mem_cons_self
      · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ e)

theorem pickF_max : ∀ (m : Cand) (l : List Cand) (y : Cand), y ∈ m :: l → ¬ Better y (pickF m l)
  | m, [], y, hy => by
    simp only [pickF]
    rcases List.mem_cons.mp hy with e | e
    · rw [e]; exact better_irrefl _
    · simp at e
  | m, x :: xs, y, hy => by
    simp only [pickF]
    have ih := pickF_max (if Better x m then x else m) xs
    have hm := ih _ List.mem_cons_self
    rcases List.mem_cons.mp hy with e | hy'
    · rw [e]
      by_cases h : Better x m
      · rw [if_pos h] at hm ⊢
        intro hyr; exact hm (better_trans h hyr)
      · rw [if_neg h] at hm ⊢; exact hm
    · rcases List.mem_cons.mp hy' with e | hy''
      · rw [e]
        by_cases h : Better x m
        · rw [if_pos h] at hm ⊢; exact hm
        · rw [if_neg h] at hm ⊢; exact not_better_trans h hm
      · exact ih y (List.mem_cons_of_mem _ hy'')

theorem best_mem {l : List Cand} {r : Cand} (h : best l = some r) : r ∈ l := by
  cases l with
  | nil => simp [best] at h
  | cons c cs => simp only [best, Option.some.injEq] at h; rw [← h]; exact pickF_mem c cs

theorem best_max {l : List Cand} {r : Cand} (h : best l = some r) : ∀ y ∈ l, ¬ Better y r := by
  cases l with
  | nil => simp [best] at h
  | cons c cs => simp only [best, Option.some.injEq] at h; rw [← h]; exact pickF_max c cs

theorem best_none {l : List Cand} : best l = none ↔ l = [] := by
  cases l <;> simp [best]

/-- Two unbeaten candidates are the same candidate. -/
theorem max_unique {r r' : Cand} (h1 : ¬ Better r' r) (h2 : ¬ Better r r') : r = r' := by
  obtain ⟨a1, a2⟩ := r; obtain ⟨b1, b2⟩ := r'
  simp only [Better] at h1 h2
  have hs : a2 = b2 := by omega
  have ht : a1 = b1 := by omega
  rw [hs, ht]

/-- The pick is determined by the set of candidates: any reordering
    gives the same pick. -/
theorem best_perm {l l' : List Cand} (hp : l.Perm l') : best l = best l' := by
  cases h : best l with
  | none =>
    have e : l = [] := best_none.mp h
    subst e
    have e' : l' = [] := (List.Perm.nil_eq hp).symm
    subst e'; rfl
  | some r =>
    have hl' : l' ≠ [] := by
      intro e; subst e
      have := List.Perm.eq_nil hp; subst this; simp [best] at h
    cases h' : best l' with
    | none => exact absurd (best_none.mp h') hl'
    | some r' =>
      have hr : r ∈ l' := hp.subset (best_mem h)
      have hr' : r' ∈ l := hp.symm.subset (best_mem h')
      rw [max_unique (best_max h r' hr') (best_max h' r hr)]

/-! ### Certificate -/

def omax : Option Int → Int → Int
  | none, b => b
  | some a, b => max a b

/-- Highest score among candidates whose token differs from `v`. -/
def rivalMax (v : Nat) : List Cand → Option Int
  | [] => none
  | y :: ys => if y.1 = v then rivalMax v ys else some (omax (rivalMax v ys) y.2)

theorem rivalMax_ge : ∀ (v : Nat) (l : List Cand) (y : Cand), y ∈ l → y.1 ≠ v →
    ∃ a, rivalMax v l = some a ∧ y.2 ≤ a
  | v, [], y, hy, _ => by simp at hy
  | v, x :: xs, y, hy, hne => by
    rcases List.mem_cons.mp hy with e | hy'
    · subst e
      simp only [rivalMax, if_neg hne]
      refine ⟨_, rfl, ?_⟩
      cases rivalMax v xs <;> simp only [omax] <;> omega
    · obtain ⟨a, ha, hle⟩ := rivalMax_ge v xs y hy' hne
      by_cases hx : x.1 = v
      · simp only [rivalMax, if_pos hx]; exact ⟨a, ha, hle⟩
      · simp only [rivalMax, if_neg hx, ha, omax]; exact ⟨_, rfl, by omega⟩

theorem rivalMax_some : ∀ (v : Nat) (l : List Cand) (a : Int), rivalMax v l = some a →
    ∃ y ∈ l, y.1 ≠ v ∧ y.2 = a
  | v, [], a, h => by simp [rivalMax] at h
  | v, x :: xs, a, h => by
    by_cases hx : x.1 = v
    · simp only [rivalMax, if_pos hx] at h
      obtain ⟨y, hy, h1, h2⟩ := rivalMax_some v xs a h
      exact ⟨y, List.mem_cons_of_mem _ hy, h1, h2⟩
    · simp only [rivalMax, if_neg hx, Option.some.injEq] at h
      cases hr : rivalMax v xs with
      | none =>
        rw [hr] at h; simp only [omax] at h
        exact ⟨x, List.mem_cons_self, hx, h⟩
      | some b =>
        rw [hr] at h; simp only [omax] at h
        by_cases hb : b ≤ x.2
        · exact ⟨x, List.mem_cons_self, hx, by omega⟩
        · obtain ⟨y, hy, h1, h2⟩ := rivalMax_some v xs b hr
          exact ⟨y, List.mem_cons_of_mem _ hy, h1, by omega⟩

/-- A certificate: the pick, its score, the margin over the best rival
    (none when there is no rival). -/
structure Cert where
  tok : Nat
  score : Int
  margin : Option Int
  deriving Repr, DecidableEq

def Cert.certified (c : Cert) : Bool :=
  match c.margin with
  | none => true
  | some m => decide (0 < m)

def certify (l : List Cand) : Option Cert :=
  match best l with
  | none => none
  | some r => some ⟨r.1, r.2, (rivalMax r.1 l).map (fun a => r.2 - a)⟩

theorem certify_spec {l : List Cand} {c : Cert} (h : certify l = some c) :
    ∃ r, best l = some r ∧ c = ⟨r.1, r.2, (rivalMax r.1 l).map (fun a => r.2 - a)⟩ := by
  unfold certify at h
  cases hb : best l with
  | none => rw [hb] at h; simp at h
  | some r => rw [hb] at h; simp only [Option.some.injEq] at h; exact ⟨r, rfl, h.symm⟩

/-- Soundness: a certified pick strictly beats every other token, by at
    least the margin. Nothing left can beat or tie it. -/
theorem certify_sound {l : List Cand} {c : Cert} (h : certify l = some c)
    (hc : c.certified = true) :
    (c.tok, c.score) ∈ l ∧ ∀ y ∈ l, y.1 ≠ c.tok → y.2 < c.score ∧
      ∀ m, c.margin = some m → y.2 + m ≤ c.score := by
  obtain ⟨r, hb, rfl⟩ := certify_spec h
  refine ⟨best_mem hb, ?_⟩
  intro y hy hne
  obtain ⟨a, ha, hle⟩ := rivalMax_ge r.1 l y hy hne
  simp only [Cert.certified, ha, Option.map_some', decide_eq_true_eq] at hc
  simp only [ha, Option.map_some', Option.some.injEq]
  refine ⟨by omega, ?_⟩
  intro m hm; omega

/-- Completeness: a candidate that strictly beats every other token is
    picked and certified. -/
theorem certify_complete {l : List Cand} {x : Cand} (hx : x ∈ l)
    (hbeat : ∀ y ∈ l, y.1 ≠ x.1 → y.2 < x.2) (hdup : ∀ y ∈ l, y.1 = x.1 → y.2 = x.2) :
    ∃ c, certify l = some c ∧ c.tok = x.1 ∧ c.score = x.2 ∧ c.certified = true := by
  cases hb : best l with
  | none => rw [best_none] at hb; subst hb; simp at hx
  | some r =>
    have hr := best_mem hb
    have hmax := best_max hb x hx
    have hrx : r.1 = x.1 := by
      by_cases hne : r.1 = x.1
      · exact hne
      · exfalso
        have := hbeat r hr hne
        apply hmax; simp only [Better]; omega
    have hs : r.2 = x.2 := hdup r hr hrx
    refine ⟨⟨r.1, r.2, (rivalMax r.1 l).map (fun a => r.2 - a)⟩, by simp [certify, hb], hrx, hs, ?_⟩
    simp only [Cert.certified]
    cases ha : rivalMax r.1 l with
    | none => simp
    | some a =>
      obtain ⟨y, hy, hne, hya⟩ := rivalMax_some r.1 l a ha
      have := hbeat y hy (by rw [← hrx]; exact hne)
      simp only [Option.map_some', decide_eq_true_eq]; omega

/-! ### Named axes, reweighting, the cone -/

/-- Weighted sum of features (axes beyond either list contribute 0). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

def vadd : List Int → List Int → List Int
  | a :: as, b :: bs => (a + b) :: vadd as bs
  | _, _ => []

theorem dot_vadd : ∀ (t d f : List Int), t.length = d.length →
    dot (vadd t d) f = dot t f + dot d f
  | [], [], f, _ => by simp [vadd, dot]
  | a :: as, b :: bs, [], _ => by simp [vadd, dot]
  | a :: as, b :: bs, c :: cs, h => by
    simp only [vadd, dot]
    rw [dot_vadd as bs cs (by simpa using h), Int.add_mul]
    generalize a * c = p; generalize b * c = q; generalize dot as cs = u; generalize dot bs cs = w
    omega
  | [], _ :: _, _, h => by simp at h
  | _ :: _, [], _, h => by simp at h

/-- Scores of the candidate edges under weights `θ`; `feat t` is the
    axis vector (the edge's colors) of token `t`. -/
def scores (base : List Cand) (feat : Nat → List Int) (θ : List Int) : List Cand :=
  base.map (fun c => (c.1, c.2 + dot θ (feat c.1)))

/-- Move every score by a per-token amount. -/
def shift (δ : Nat → Int) (l : List Cand) : List Cand := l.map (fun c => (c.1, c.2 + δ c.1))

/-- Reweight an already scored list by a weight change `Δ`. -/
def rescore (Δ : List Int) (feat : Nat → List Int) (l : List Cand) : List Cand :=
  shift (fun t => dot Δ (feat t)) l

/-- Reweighting equals recompute. -/
theorem rescore_eq (base : List Cand) (feat : Nat → List Int) (θ Δ : List Int)
    (h : θ.length = Δ.length) :
    rescore Δ feat (scores base feat θ) = scores base feat (vadd θ Δ) := by
  simp only [rescore, shift, scores, List.map_map]
  apply List.map_congr_left
  intro c _
  simp only [Function.comp, dot_vadd θ Δ _ h]
  congr 1; omega

theorem mem_shift {δ : Nat → Int} {l : List Cand} {y : Cand} :
    y ∈ shift δ l ↔ ∃ x ∈ l, y = (x.1, x.2 + δ x.1) := by
  simp only [shift, List.mem_map]; constructor
  · rintro ⟨x, hx, rfl⟩; exact ⟨x, hx, rfl⟩
  · rintro ⟨x, hx, rfl⟩; exact ⟨x, hx, rfl⟩

/-- The cone, sufficient form: if the pick is certified with margin `m`
    and every score moves by less than `m/2` in absolute value, the pick
    is unchanged. Checking this needs only the margin and the moves, no
    recompute of the base scores. Assumes one candidate per token. -/
theorem stable_of_bound {l : List Cand} {c : Cert} {m : Int} (δ : Nat → Int)
    (h : certify l = some c) (hm : c.margin = some m) (hpos : 0 < m)
    (hb : ∀ y ∈ l, -m < 2 * δ y.1 ∧ 2 * δ y.1 < m) :
    ∃ r, best (shift δ l) = some r ∧ r.1 = c.tok := by
  have hc : c.certified = true := by simp [Cert.certified, hm, hpos]
  obtain ⟨hmem, hbeat⟩ := certify_sound h hc
  have hx : (c.tok, c.score + δ c.tok) ∈ shift δ l := mem_shift.mpr ⟨_, hmem, rfl⟩
  cases hr : best (shift δ l) with
  | none => rw [best_none] at hr; simp [hr] at hx
  | some r =>
    refine ⟨r, rfl, ?_⟩
    obtain ⟨y, hy, rfl⟩ := mem_shift.mp (best_mem hr)
    by_cases hne' : y.1 = c.tok
    · exact hne'
    exfalso
    have hne : y.1 ≠ c.tok := hne'
    have hmax := best_max hr _ hx
    obtain ⟨_, hmarg⟩ := hbeat y hy hne
    have h1 := hmarg m hm
    have h2 := hb y hy
    have h3 := hb _ hmem
    apply hmax
    simp only [Better]; left
    dsimp only at h3 ⊢
    omega

/-- Setting one weight. -/
def setAt : List Int → Nat → Int → List Int
  | [], _, _ => []
  | _ :: as, 0, x => x :: as
  | a :: as, n + 1, x => a :: setAt as n x

theorem setAt_length : ∀ (t : List Int) (n : Nat) (x : Int), (setAt t n x).length = t.length
  | [], _, _ => rfl
  | _ :: _, 0, _ => rfl
  | _ :: as, n + 1, x => by simp [setAt, setAt_length as n x]

/-- An axis whose feature is zero does not change the weighted sum,
    whatever its weight. -/
theorem dot_set_zero : ∀ (t f : List Int) (a : Nat) (x : Int), f.getD a 0 = 0 →
    dot (setAt t a x) f = dot t f
  | [], f, a, x, _ => by simp [setAt, dot]
  | b :: bs, [], a, x, _ => by cases a <;> simp [setAt, dot]
  | b :: bs, c :: cs, 0, x, h => by simp at h; simp [setAt, dot, h]
  | b :: bs, c :: cs, n + 1, x, h => by
    rw [List.getD_cons_succ] at h
    simp only [setAt, dot]; rw [dot_set_zero bs cs n x h]

/-- Parameter locality at one step. -/
theorem scores_local (base : List Cand) (feat : Nat → List Int) (θ : List Int) (a : Nat) (x : Int)
    (hz : ∀ c ∈ base, (feat c.1).getD a 0 = 0) :
    scores base feat (setAt θ a x) = scores base feat θ := by
  simp only [scores]
  apply List.map_congr_left
  intro c hc; rw [dot_set_zero θ (feat c.1) a x (hz c hc)]

/-! ### Generation -/

/-- A model: base candidate scores and axis features, both functions of
    the content (the token prefix). Nothing else is assumed about it. -/
structure Model where
  base : List Nat → List Cand
  feat : List Nat → Nat → List Int

def step (M : Model) (θ : List Int) (c : List Nat) : Option Nat :=
  (best (scores (M.base c) (M.feat c) θ)).map Prod.fst

def gen (M : Model) (θ : List Int) : Nat → List Nat → List Nat
  | 0, c => c
  | n + 1, c =>
    match step M θ c with
    | none => c
    | some v => gen M θ n (c ++ [v])

theorem gen_prefix (M : Model) (θ : List Int) : ∀ (n : Nat) (c : List Nat), c <+: gen M θ n c
  | 0, c => List.prefix_refl c
  | n + 1, c => by
    simp only [gen]
    cases step M θ c with
    | none => exact List.prefix_refl c
    | some v => exact (List.prefix_append c [v]).trans (gen_prefix M θ n (c ++ [v]))

theorem scores_perm {base base' : List Cand} (feat : Nat → List Int) (θ : List Int)
    (hp : base.Perm base') : (scores base feat θ).Perm (scores base' feat θ) :=
  hp.map _

/-- Generation does not depend on the order in which candidates are
    enumerated. -/
theorem gen_perm (M M' : Model) (θ : List Int) (hf : M.feat = M'.feat)
    (hp : ∀ c, (M.base c).Perm (M'.base c)) : ∀ n c, gen M θ n c = gen M' θ n c
  | 0, c => rfl
  | n + 1, c => by
    have hs : step M θ c = step M' θ c := by
      simp only [step]; rw [hf, best_perm (scores_perm _ _ (hp c))]
    simp only [gen, hs]
    cases step M' θ c with
    | none => rfl
    | some v => exact gen_perm M M' θ hf hp n (c ++ [v])

/-- Parameter locality for generation: an axis that is zero on every
    content leaves every output unchanged, whatever its weight. -/
theorem gen_local (M : Model) (θ : List Int) (a : Nat) (x : Int)
    (hz : ∀ c t, (M.feat c t).getD a 0 = 0) :
    ∀ n c, gen M (setAt θ a x) n c = gen M θ n c
  | 0, c => rfl
  | n + 1, c => by
    have hs : step M (setAt θ a x) c = step M θ c := by
      simp only [step]; rw [scores_local _ _ θ a x (fun cc _ => hz c cc.1)]
    simp only [gen, hs]
    cases step M θ c with
    | none => rfl
    | some v => exact gen_local M θ a x hz n (c ++ [v])

/-- A cache of base scores keyed by content. -/
def lookup (cache : List (List Nat × List Cand)) (c : List Nat) : Option (List Cand) :=
  (cache.find? (fun e => e.1 == c)).map Prod.snd

def cached (M : Model) (cache : List (List Nat × List Cand)) : Model :=
  { M with base := fun c => (lookup cache c).getD (M.base c) }

theorem lookup_correct (M : Model) (cache : List (List Nat × List Cand))
    (hok : ∀ e ∈ cache, e.2 = M.base e.1) (c : List Nat) :
    (lookup cache c).getD (M.base c) = M.base c := by
  unfold lookup
  cases h : cache.find? (fun e => e.1 == c) with
  | none => rfl
  | some e =>
    have hm := List.mem_of_find?_eq_some h
    have he : e.1 = c := by have := List.find?_some h; simpa using this
    simp only [Option.map_some', Option.getD_some]; rw [hok e hm, he]

/-- A correct cache changes nothing. -/
theorem gen_cache (M : Model) (cache : List (List Nat × List Cand))
    (hok : ∀ e ∈ cache, e.2 = M.base e.1) (θ : List Int) (n : Nat) (c : List Nat) :
    gen (cached M cache) θ n c = gen M θ n c := by
  have : cached M cache = M := by
    cases M with
    | mk b f =>
      simp only [cached, Model.mk.injEq, and_true]
      funext c'; exact lookup_correct ⟨b, f⟩ cache hok c'
  rw [this]

/-- One reweighted step computed from the scored list of the old
    weights, without touching the base scores. -/
def stepRe (M : Model) (θ Δ : List Int) (c : List Nat) : Option Nat :=
  (best (rescore Δ (M.feat c) (scores (M.base c) (M.feat c) θ))).map Prod.fst

theorem stepRe_eq (M : Model) (θ Δ : List Int) (h : θ.length = Δ.length) (c : List Nat) :
    stepRe M θ Δ c = step M (vadd θ Δ) c := by
  simp only [stepRe, step, rescore_eq _ _ θ Δ h]

def genRe (M : Model) (θ Δ : List Int) : Nat → List Nat → List Nat
  | 0, c => c
  | n + 1, c =>
    match stepRe M θ Δ c with
    | none => c
    | some v => genRe M θ Δ n (c ++ [v])

/-- Reweighting equals recompute, for whole generations. -/
theorem gen_reweight (M : Model) (θ Δ : List Int) (h : θ.length = Δ.length) :
    ∀ n c, genRe M θ Δ n c = gen M (vadd θ Δ) n c
  | 0, c => rfl
  | n + 1, c => by
    simp only [genRe, gen, stepRe_eq M θ Δ h c]
    cases step M (vadd θ Δ) c with
    | none => rfl
    | some v => exact gen_reweight M θ Δ h n (c ++ [v])

end Prlm
