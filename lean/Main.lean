/-
  `prlm decide FILE`: replays recorded decoding steps through the proved
  definitions in Prlm.Core (scores, certify, rescore, best, vadd,
  stable_of_bound's test) and checks the harness's picks against them.

  `prlm front FILE`: Pareto front (AlgoSearch.arch, proved mem_arch) of
  minimization vectors, one per line as `name v1 v2 ...`, and for each
  front member the Chebyshev score under weights aimed at it
  (AlgoSearch.cheb, proved cheb_target), with the minimizer of that
  scalarization over all rows.

  Step record (little-endian int32 throughout):
    n k, base[n], theta[k], delta[k], nf, nf x (tok axis val),
    harness pick under theta, harness pick under theta+delta (recomputed).
-/
import Prlm.Core
import Prlm.AlgoSearch
import Prlm.Sparse
import Prlm.Robust
set_option warningAsError true

open Prlm

def u32 (b : ByteArray) (o : Nat) : Nat :=
  (b.get! o).toNat + (b.get! (o+1)).toNat * 256 + (b.get! (o+2)).toNat * 65536 +
    (b.get! (o+3)).toNat * 16777216

def i32 (b : ByteArray) (o : Nat) : Int :=
  let u := u32 b o
  if u < 2147483648 then (u : Int) else (u : Int) - 4294967296

def readI32s (b : ByteArray) (o n : Nat) : List Int :=
  (List.range n).map (fun i => i32 b (o + 4 * i))

/-- Dense axis vectors built from sparse (tok, axis, val) entries. -/
def buildFeat (n k : Nat) (entries : List (Nat × Nat × Int)) : Array (List Int) :=
  entries.foldl (fun arr (e : Nat × Nat × Int) =>
    let (t, a, v) := e
    let cur := arr.getD t []
    let cur := if cur.isEmpty then List.replicate k 0 else cur
    arr.set! t (setAt cur a (cur.getD a 0 + v))) (Array.replicate n [])

structure Tally where
  steps : Nat := 0
  agree : Nat := 0
  agreeRe : Nat := 0
  reEqRec : Nat := 0
  certified : Nat := 0
  boundStable : Nat := 0
  boundStableRight : Nat := 0
  changed : Nat := 0
  sparseEq : Nat := 0
  support : Nat := 0
  work : Nat := 0
  workMax : Nat := 0

def showOpt : Option Int → String
  | none => "none"
  | some m => toString m

partial def loop (b : ByteArray) (o : Nat) (t : Tally) (verbose reportRadius : Bool) : IO Tally := do
  if o == b.size then
    if t.steps == 0 then
      throw (IO.userError "empty replay")
    return t
  if o + 8 > b.size then
    throw (IO.userError s!"truncated replay header at byte {o}")
  let n := u32 b o
  let k := u32 b (o + 4)
  let o1 := o + 8
  let o2 := o1 + 4 * n
  let o3 := o2 + 8 * k
  if n == 0 || o3 + 4 > b.size then
    throw (IO.userError s!"invalid replay dimensions at byte {o}")
  let nf := u32 b o3
  let o4 := o3 + 4 + 12 * nf
  if o4 + 8 > b.size then
    throw (IO.userError s!"truncated replay features or picks at byte {o}")
  let baseS := readI32s b o1 n
  let base : List Cand := (List.range n).zip baseS
  let θ := readI32s b o2 k
  let Δ := readI32s b (o2 + 4 * k) k
  let entries : List (Int × Int × Int) := (List.range nf).map fun j =>
    let p := o3 + 4 + 12 * j
    (i32 b p, i32 b (p + 4), i32 b (p + 8))
  if entries.any (fun e => decide (e.1 < 0 ∨ (n : Int) ≤ e.1 ∨ e.2.1 < 0 ∨ (k : Int) ≤ e.2.1)) then
    throw (IO.userError s!"out of range feature at byte {o}")
  let ents := entries.map (fun e => (e.1.toNat, e.2.1.toNat, e.2.2))
  let hp := i32 b o4
  let hpRe := i32 b (o4 + 4)
  if decide (hp < 0 ∨ (n : Int) ≤ hp ∨ hpRe < 0 ∨ (n : Int) ≤ hpRe) then
    throw (IO.userError s!"out of range pick at byte {o}")
  let farr := buildFeat n k ents
  let feat : Nat → List Int := fun tk => farr.getD tk []
  let l := scores base feat θ
  let c := certify l
  let re := best (rescore Δ feat l)
  let retuned := scores base feat (vadd θ Δ)
  let rcm := best retuned
  let pick : Int := match c with | some c => (c.tok : Int) | none => -1
  let pRe : Int := match re with | some r => (r.1 : Int) | none => -1
  let pRec : Int := match rcm with | some r => (r.1 : Int) | none => -1
  let cert := match c with | some c => c.certified | none => false
  -- the cone test of stable_of_bound: decided from the margin and the moves only
  let δ : Nat → Int := fun tk => dot Δ (feat tk)
  let bound : Bool := match c with
    | some ⟨_, _, some m⟩ => decide (0 < m) && l.all (fun y => decide (-m < 2 * δ y.1 ∧ 2 * δ y.1 < m))
    | _ => false
  -- exact sparse reweighting (best_split): moved candidates plus the best
  -- unmoved one; work = |S| + rank of the first unmoved candidate in the
  -- old order (the order is computed once per content and reused)
  let inS : Nat → Bool := fun tk => δ tk != 0
  let moved := l.filter (fun c => inS c.1)
  let sp := best (shift δ moved ++ (best (l.filter (fun c => !inS c.1))).toList)
  let pSp : Int := match sp with | some r => (r.1 : Int) | none => -1
  let order := l.mergeSort (fun a b => decide (Better a b) || decide (a = b))
  let rank := (order.findIdx? (fun c => !inS c.1)).getD order.length
  let w := moved.length + rank + 1
  let t := { t with
    sparseEq := t.sparseEq + (if pSp == pRec then 1 else 0),
    support := t.support + moved.length,
    work := t.work + w,
    workMax := max t.workMax w,
    steps := t.steps + 1,
    agree := t.agree + (if pick == hp then 1 else 0),
    agreeRe := t.agreeRe + (if pRe == hpRe then 1 else 0),
    reEqRec := t.reEqRec + (if pRe == pRec then 1 else 0),
    certified := t.certified + (if cert then 1 else 0),
    boundStable := t.boundStable + (if bound then 1 else 0),
    boundStableRight := t.boundStableRight + (if bound && pRe == pick then 1 else 0),
    changed := t.changed + (if pRe != pick then 1 else 0) }
  if verbose then
    let m := match c with | some c => showOpt c.margin | none => "none"
    IO.println s!"step {t.steps} pick {pick} harness {hp} margin {m} certified {cert} re {pRe} rec {pRec} harnessRe {hpRe} bound {bound}"
  if reportRadius then
    let rob := (best l).bind (fun w => Robust.radius w l)
    let robRe := (best retuned).bind (fun w => Robust.radius w retuned)
    IO.println s!"radius {t.steps} {showOpt rob} {showOpt robRe}"
  loop b (o4 + 8) t verbose reportRadius

def parseVecs (s : String) : List (String × List Nat) :=
  (s.splitOn "\n").filterMap fun line =>
    match (line.trim.splitOn " ").filter (· ≠ "") with
    | [] => none
    | name :: vs => some (name, vs.map String.toNat!)

def main (args : List String) : IO UInt32 := do
  match args with
  | ["decide", f] | ["decide", f, "-v"] | ["decide", f, "--robust"] | ["decide", f, "--expect", _] =>
    let b ← IO.FS.readBinFile f
    let t ← loop b 0 {} (args.contains "-v") (args.contains "--robust")
    if let ["decide", _, "--expect", expected] := args then
      match expected.toNat? with
      | none =>
        IO.eprintln "invalid expected replay step count"
        return 2
      | some count =>
        if count != t.steps then
          IO.eprintln s!"expected {count} replay steps but found {t.steps}"
          return 1
    IO.println s!"steps {t.steps} agree {t.agree} agreeRe {t.agreeRe} reEqRec {t.reEqRec} certified {t.certified} boundStable {t.boundStable} boundStableRight {t.boundStableRight} changed {t.changed} sparseEq {t.sparseEq} support {t.support} work {t.work} workMax {t.workMax}"
    return (if t.agree == t.steps && t.agreeRe == t.steps && t.reEqRec == t.steps && t.sparseEq == t.steps then 0 else 1)
  | ["front", f] =>
    let rows := parseVecs (← IO.FS.readFile f)
    let vecs := rows.map Prod.snd
    let fr := AlgoSearch.arch vecs.reverse
    let names := rows.filter (fun r => fr.contains r.2)
    let dim := match vecs with | v :: _ => v.length | [] => 0
    -- reference point: componentwise minimum; rows are shifted up by one
    -- so the reference lies strictly below every row (cheb needs `below`)
    let zz := (List.range dim).map fun i => (vecs.map (fun v => v.getD i 0)).foldl min (vecs.headD [] |>.getD i 0)
    let sh : List Nat → List Nat := fun v => v.map (· + 1)
    for (nm, f0) in names do
      let fv := sh f0
      let scored := rows.map fun r => (r.1, AlgoSearch.cheb 1000000 fv zz (sh r.2))
      let mn := scored.foldl (fun a s => min a s.2) (scored.headD ("", 0)).2
      let winners := (scored.filter (·.2 == mn)).map Prod.fst
      IO.println s!"front {nm} {f0} chebmin {winners}"
    return 0
  | _ =>
    IO.println "usage: prlm decide FILE [-v | --robust | --expect COUNT] | prlm front FILE"
    return 2
