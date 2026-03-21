import lean.Basic

lemma rectangleFree_1xn {k n : ℕ} (A : Coloring k 1 n) : RectangleFree A := by
  intro h
  rcases h with ⟨r₁, r₂, c₁, c₂, hr, _, _, _, _⟩
  apply hr
  simpa using (Fin.eq_zero r₁).trans (Fin.eq_zero r₂).symm

theorem T_1xn (k n : ℕ) : T k 1 n = k ^ n := by
  let e : {A : Coloring k 1 n // RectangleFree A} ≃ Coloring k 1 n :=
    { toFun := Subtype.val
      invFun := fun A => ⟨A, rectangleFree_1xn A⟩
      left_inv := by
        intro x
        apply Subtype.ext
        rfl
      right_inv := by
        intro A
        rfl }
  simpa [T, Coloring, Fintype.card_fun, Fintype.card_fin] using
    Fintype.card_congr e
