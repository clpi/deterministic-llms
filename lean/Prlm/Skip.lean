/-
  Forward skipping by drafted continuations (harness/skip.py, verified
  mode). A draft is a continuation recorded from a witness history; the
  verifier accepts drafted tokens while each equals the certified pick.

  Proved:
  * `accept_gen`: the accepted tokens are exactly the next tokens of the
    rule's own generation. Skipping changes the number of model calls,
    never the output.
  * `accept_batch`: if the verifier's scores (one forward over content
    plus draft) equal the per-step scores, it accepts the same tokens.
    That equality is a PREMISE about floating point, not proved here; it
    is measured (bitwise) in the run.
-/
import Prlm.Core

namespace Prlm

/-- Accept drafted tokens while each equals the rule's pick. -/
def accept (M : Model) (θ : List Int) : List Nat → List Nat → List Nat
  | _, [] => []
  | c, v :: vs => if step M θ c = some v then v :: accept M θ (c ++ [v]) vs else []

theorem accept_gen (M : Model) (θ : List Int) :
    ∀ (d c : List Nat), c ++ accept M θ c d = gen M θ (accept M θ c d).length c
  | [], c => by simp [accept, gen]
  | v :: vs, c => by
    by_cases h : step M θ c = some v
    · have ih := accept_gen M θ vs (c ++ [v])
      simp only [accept, h, if_true, List.length_cons, gen]
      rw [← ih]; simp
    · simp [accept, h, gen]

theorem accept_batch (M Mb : Model) (θ : List Int) (hb : ∀ c, Mb.base c = M.base c)
    (hf : Mb.feat = M.feat) (c d : List Nat) : accept Mb θ c d = accept M θ c d := by
  have : Mb = M := by
    cases Mb; cases M
    simp only [Model.mk.injEq] at hf ⊢
    exact ⟨funext hb, hf⟩
  rw [this]

end Prlm
