import Lake
open Lake DSL

package prlm

lean_lib Prlm where
  roots := #[`Prlm.Core, `Prlm.AlgoSearch, `Prlm.Skip, `Prlm.Sparse, `Prlm.Loop, `Prlm.Fix, `Prlm.Struct, `Prlm.Robust]

@[default_target]
lean_exe prlm where
  root := `Main
