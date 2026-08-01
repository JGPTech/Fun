import Lake

open Lake DSL

package degree_six_keller where
  version := v!"0.1.0"

require "leanprover-community" / "mathlib" @ git
  "905b95818eb32af7874a58b427f50c1711a5e96c"

@[default_target]
lean_lib DegreeSixKeller
