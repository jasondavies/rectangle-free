import Mathlib

open Classical

abbrev Coloring (k m n : ℕ) := Fin m → Fin n → Fin k

def RectangleFree {k m n : ℕ} (A : Coloring k m n) : Prop :=
  ¬ ∃ r₁ r₂ c₁ c₂,
      r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧
      A r₁ c₁ = A r₁ c₂ ∧
      A r₁ c₁ = A r₂ c₁ ∧
      A r₁ c₁ = A r₂ c₂

noncomputable def T (k m n : ℕ) : ℕ :=
  Fintype.card {A : Coloring k m n // RectangleFree A}
