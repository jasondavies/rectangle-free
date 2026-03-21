import lean.Basic

open Classical

abbrev MixedColumn (k : ℕ) := {p : Fin k × Fin k // p.1 ≠ p.2}

def monoSet {k n : ℕ} (A : Coloring k 2 n) : Finset (Fin n) :=
  Finset.univ.filter fun c => A 0 c = A 1 c

lemma mem_monoSet_iff {k n : ℕ} (A : Coloring k 2 n) (c : Fin n) :
    c ∈ monoSet A ↔ A 0 c = A 1 c := by
  simp [monoSet]

abbrev RectFiber (k n : ℕ) (S : Finset (Fin n)) :=
  {A : Coloring k 2 n // RectangleFree A ∧ monoSet A = S}

def monoFiberEquiv {k n : ℕ} (S : Finset (Fin n)) :
    {A : {A : Coloring k 2 n // RectangleFree A} // monoSet A.1 = S} ≃ RectFiber k n S :=
  { toFun := fun A => ⟨A.1.1, A.1.2, A.2⟩
    invFun := fun A => ⟨⟨A.1, A.2.1⟩, A.2.2⟩
    left_inv := by
      intro A
      apply Subtype.ext
      apply Subtype.ext
      rfl
    right_inv := by
      intro A
      apply Subtype.ext
      rfl }

def monoEmbedding {k n : ℕ} (A : Coloring k 2 n) (hA : RectangleFree A)
    (S : Finset (Fin n)) (hS : monoSet A = S) : ↥S ↪ Fin k :=
  { toFun := fun c => A 0 c
    inj' := by
      intro c₁ c₂ hcol
      apply Subtype.ext
      by_contra hne
      have hc₁ : A 0 c₁ = A 1 c₁ := by
        have : (c₁ : Fin n) ∈ monoSet A := by
          rw [hS]
          exact c₁.2
        exact (mem_monoSet_iff A c₁).1 this
      have hc₂ : A 0 c₂ = A 1 c₂ := by
        have : (c₂ : Fin n) ∈ monoSet A := by
          rw [hS]
          exact c₂.2
        exact (mem_monoSet_iff A c₂).1 this
      exact hA ⟨0, 1, c₁, c₂, by decide, hne, hcol, hc₁, hcol.trans hc₂⟩ }

def mixedFunction {k n : ℕ} (A : Coloring k 2 n) (S : Finset (Fin n))
    (hS : monoSet A = S) : ↥(Sᶜ) → MixedColumn k := fun c =>
  ⟨(A 0 c, A 1 c), by
    intro hEq
    have hnot : (c : Fin n) ∉ S := Finset.mem_compl.mp c.2
    have hmem : (c : Fin n) ∈ S := by
      have : (c : Fin n) ∈ monoSet A := (mem_monoSet_iff A c).2 hEq
      simpa [hS] using this
    exact hnot hmem⟩

abbrev RectData (k n : ℕ) (S : Finset (Fin n)) :=
  (↥S ↪ Fin k) × (↥(Sᶜ) → MixedColumn k)

def buildColoring {k n : ℕ} (S : Finset (Fin n)) (d : RectData k n S) : Coloring k 2 n :=
  fun r c =>
    if h : c ∈ S then
      d.1 ⟨c, h⟩
    else
      let p := d.2 ⟨c, by simpa [Finset.mem_compl] using h⟩
      if r = 0 then p.1.1 else p.1.2

lemma monoSet_buildColoring {k n : ℕ} (S : Finset (Fin n)) (d : RectData k n S) :
    monoSet (buildColoring S d) = S := by
  ext c
  by_cases h : c ∈ S
  · simp [mem_monoSet_iff, buildColoring, h]
  · have hneq : buildColoring S d 0 c ≠ buildColoring S d 1 c := by
      simpa [buildColoring, h] using
        (d.2 ⟨c, by simpa [Finset.mem_compl] using h⟩).2
    simp [mem_monoSet_iff, h, hneq]

lemma rectangleFree_buildColoring {k n : ℕ} (S : Finset (Fin n)) (d : RectData k n S) :
    RectangleFree (buildColoring S d) := by
  intro h
  rcases h with ⟨r₁, r₂, c₁, c₂, hr, hc, hrow, hcol₁, hcol₂⟩
  fin_cases r₁ <;> fin_cases r₂
  · cases hr rfl
  ·
    have hc₁S : c₁ ∈ S := by
      have : c₁ ∈ monoSet (buildColoring S d) := (mem_monoSet_iff _ _).2 hcol₁
      simpa [monoSet_buildColoring] using this
    have hc₂eq : buildColoring S d 0 c₂ = buildColoring S d 1 c₂ := by
      calc
        buildColoring S d 0 c₂ = buildColoring S d 0 c₁ := hrow.symm
        _ = buildColoring S d 1 c₂ := hcol₂
    have hc₂S : c₂ ∈ S := by
      have : c₂ ∈ monoSet (buildColoring S d) := (mem_monoSet_iff _ _).2 hc₂eq
      simpa [monoSet_buildColoring] using this
    have hEq : d.1 ⟨c₁, hc₁S⟩ = d.1 ⟨c₂, hc₂S⟩ := by
      simpa [buildColoring, hc₁S, hc₂S] using hrow
    exact hc (congrArg Subtype.val (d.1.injective hEq))
  ·
    have hc₁S : c₁ ∈ S := by
      have : c₁ ∈ monoSet (buildColoring S d) := (mem_monoSet_iff _ _).2 hcol₁.symm
      simpa [monoSet_buildColoring] using this
    have hc₂eq : buildColoring S d 0 c₂ = buildColoring S d 1 c₂ := by
      calc
        buildColoring S d 0 c₂ = buildColoring S d 1 c₁ := hcol₂.symm
        _ = buildColoring S d 1 c₂ := hrow
    have hc₂S : c₂ ∈ S := by
      have : c₂ ∈ monoSet (buildColoring S d) := (mem_monoSet_iff _ _).2 hc₂eq
      simpa [monoSet_buildColoring] using this
    have hEq : d.1 ⟨c₁, hc₁S⟩ = d.1 ⟨c₂, hc₂S⟩ := by
      simpa [buildColoring, hc₁S, hc₂S] using hrow
    exact hc (congrArg Subtype.val (d.1.injective hEq))
  · cases hr rfl

def rectFiberEquiv {k n : ℕ} (S : Finset (Fin n)) : RectFiber k n S ≃ RectData k n S :=
  { toFun := fun A => ⟨monoEmbedding A.1 A.2.1 S A.2.2, mixedFunction A.1 S A.2.2⟩
    invFun := fun d => ⟨buildColoring S d, rectangleFree_buildColoring S d, monoSet_buildColoring S d⟩
    left_inv := by
      intro A
      apply Subtype.ext
      funext r c
      by_cases h : c ∈ S
      · have hmono : A.1 0 c = A.1 1 c := by
          have : c ∈ monoSet A.1 := by simpa [A.2.2] using h
          exact (mem_monoSet_iff A.1 c).1 this
        fin_cases r
        · simp [buildColoring, monoEmbedding, h]
        · simp [buildColoring, monoEmbedding, h, hmono]
      · fin_cases r
        · simp [buildColoring, mixedFunction, h]
        · simp [buildColoring, mixedFunction, h]
    right_inv := by
      rintro ⟨e, f⟩
      apply Prod.ext
      · ext c
        simp [monoEmbedding, buildColoring]
      · funext c
        have hnot : (c : Fin n) ∉ S := Finset.mem_compl.mp c.2
        apply Subtype.ext
        simp [mixedFunction, buildColoring, hnot] }

lemma mixedColumnCard (k : ℕ) : Fintype.card (MixedColumn k) = k ^ 2 - k := by
  let e : {p : Fin k × Fin k // p.1 = p.2} ≃ Fin k :=
    { toFun := fun p => p.1.1
      invFun := fun c => ⟨(c, c), rfl⟩
      left_inv := by
        intro p
        rcases p with ⟨⟨a, b⟩, hab⟩
        cases hab
        rfl
      right_inv := by
        intro c
        rfl }
  have hdiag : Fintype.card {p : Fin k × Fin k // p.1 = p.2} = k := by
    simpa [Fintype.card_fin] using Fintype.card_congr e
  calc
    Fintype.card (MixedColumn k)
        = Fintype.card {p : Fin k × Fin k // p.1 ≠ p.2} := by
            rfl
    _ = Fintype.card (Fin k × Fin k) - Fintype.card {p : Fin k × Fin k // p.1 = p.2} := by
            exact Fintype.card_subtype_compl (fun p : Fin k × Fin k => p.1 = p.2)
    _ = k * k - k := by rw [Fintype.card_prod, Fintype.card_fin, hdiag]
    _ = k ^ 2 - k := by simp [pow_two]

lemma rectDataCard {k n : ℕ} (S : Finset (Fin n)) :
    Fintype.card (RectData k n S) = k.descFactorial S.card * (k ^ 2 - k) ^ (n - S.card) := by
  have hEmb : Fintype.card (↥S ↪ Fin k) = k.descFactorial S.card := by
    rw [Fintype.card_embedding_eq]
    simp [Fintype.card_fin, Fintype.card_coe]
  have hMixed :
      Fintype.card (↥(Sᶜ) → MixedColumn k) = (k ^ 2 - k) ^ (n - S.card) := by
    rw [Fintype.card_fun, mixedColumnCard]
    simp [Fintype.card_coe, Fintype.card_fin]
  calc
    Fintype.card (RectData k n S)
        = Fintype.card (↥S ↪ Fin k) * Fintype.card (↥(Sᶜ) → MixedColumn k) := by
            simp [RectData, Fintype.card_prod]
    _ = k.descFactorial S.card * (k ^ 2 - k) ^ (n - S.card) := by rw [hEmb, hMixed]

lemma powersetCard_biUnion_range (n : ℕ) :
    (Finset.range (n + 1)).biUnion
        (fun s => Finset.powersetCard s (Finset.univ : Finset (Fin n)))
      = (Finset.univ : Finset (Finset (Fin n))) := by
  ext S
  constructor
  · intro _
    simp
  · intro _
    refine Finset.mem_biUnion.2 ?_
    refine ⟨S.card, Finset.mem_range.2 (Nat.lt_succ_of_le (by simpa [Fintype.card_fin] using Finset.card_le_univ S)), ?_⟩
    exact Finset.mem_powersetCard.2 ⟨Finset.subset_univ S, rfl⟩

lemma powersetCard_pairwiseDisjoint (n : ℕ) :
    (((Finset.range (n + 1) : Finset ℕ) : Set ℕ)).PairwiseDisjoint
      (fun s => Finset.powersetCard s (Finset.univ : Finset (Fin n))) := by
  intro a ha b hb hab
  show Disjoint (Finset.powersetCard a (Finset.univ : Finset (Fin n)))
    (Finset.powersetCard b (Finset.univ : Finset (Fin n)))
  rw [Finset.disjoint_left]
  intro S hSa hSb
  have hca : S.card = a := (Finset.mem_powersetCard.1 hSa).2
  have hcb : S.card = b := (Finset.mem_powersetCard.1 hSb).2
  exact hab (hca.symm.trans hcb)

lemma sum_rectData_by_card (k n : ℕ) :
    (∑ S : Finset (Fin n), k.descFactorial S.card * (k ^ 2 - k) ^ (n - S.card))
      =
    Finset.sum (Finset.range (n + 1))
      (fun s => Nat.choose n s * (k.descFactorial s * (k ^ 2 - k) ^ (n - s))) := by
  calc
    (∑ S : Finset (Fin n), k.descFactorial S.card * (k ^ 2 - k) ^ (n - S.card))
        = Finset.sum
            ((Finset.range (n + 1)).biUnion
              (fun s => Finset.powersetCard s (Finset.univ : Finset (Fin n))))
            (fun S => k.descFactorial S.card * (k ^ 2 - k) ^ (n - S.card)) := by
              rw [powersetCard_biUnion_range]
    _ = Finset.sum (Finset.range (n + 1))
          (fun s =>
            Finset.sum (Finset.powersetCard s (Finset.univ : Finset (Fin n)))
              (fun S => k.descFactorial S.card * (k ^ 2 - k) ^ (n - S.card))) := by
              exact Finset.sum_biUnion (powersetCard_pairwiseDisjoint n)
    _ = Finset.sum (Finset.range (n + 1))
          (fun s =>
            Finset.sum (Finset.powersetCard s (Finset.univ : Finset (Fin n)))
              (fun _S => k.descFactorial s * (k ^ 2 - k) ^ (n - s))) := by
              refine Finset.sum_congr rfl ?_
              intro s hs
              refine Finset.sum_congr rfl ?_
              intro S hS
              simp [(Finset.mem_powersetCard.1 hS).2]
    _ = Finset.sum (Finset.range (n + 1))
          (fun s => Nat.choose n s * (k.descFactorial s * (k ^ 2 - k) ^ (n - s))) := by
              refine Finset.sum_congr rfl ?_
              intro s hs
              have hconst :
                  Finset.sum (Finset.powersetCard s (Finset.univ : Finset (Fin n)))
                    (fun _S => k.descFactorial s * (k ^ 2 - k) ^ (n - s))
                    =
                  (Finset.powersetCard s (Finset.univ : Finset (Fin n))).card *
                    (k.descFactorial s * (k ^ 2 - k) ^ (n - s)) := by
                      exact
                        Finset.sum_const_nat
                          (s := Finset.powersetCard s (Finset.univ : Finset (Fin n)))
                          (m := k.descFactorial s * (k ^ 2 - k) ^ (n - s))
                          (fun _ _ => rfl)
              rw [hconst, Finset.card_powersetCard]
              simp [Finset.card_univ, Fintype.card_fin]

theorem T_2xn (k n : ℕ) :
    T k 2 n =
      Finset.sum (Finset.range (n + 1))
        (fun s => Nat.choose n s * k.descFactorial s * (k ^ 2 - k) ^ (n - s)) := by
  let f : {A : Coloring k 2 n // RectangleFree A} → Finset (Fin n) := fun A => monoSet A.1
  calc
    T k 2 n
        = Fintype.card ((S : Finset (Fin n)) × {A : {A : Coloring k 2 n // RectangleFree A} // f A = S}) := by
            simpa [T, f] using
              Fintype.card_congr (Equiv.sigmaFiberEquiv f).symm
    _ = ∑ S : Finset (Fin n),
          Fintype.card {A : {A : Coloring k 2 n // RectangleFree A} // f A = S} := by
            rw [Fintype.card_sigma]
    _ = ∑ S : Finset (Fin n), Fintype.card (RectFiber k n S) := by
            refine Finset.sum_congr rfl ?_
            intro S hS
            simpa [f] using Fintype.card_congr (monoFiberEquiv (k := k) (n := n) S)
    _ = ∑ S : Finset (Fin n), Fintype.card (RectData k n S) := by
            refine Finset.sum_congr rfl ?_
            intro S hS
            simpa using Fintype.card_congr (rectFiberEquiv (k := k) (n := n) S)
    _ = ∑ S : Finset (Fin n), k.descFactorial S.card * (k ^ 2 - k) ^ (n - S.card) := by
            refine Finset.sum_congr rfl ?_
            intro S hS
            exact rectDataCard (k := k) (n := n) S
    _ = Finset.sum (Finset.range (n + 1))
          (fun s => Nat.choose n s * (k.descFactorial s * (k ^ 2 - k) ^ (n - s))) := by
            exact sum_rectData_by_card k n
    _ = Finset.sum (Finset.range (n + 1))
          (fun s => Nat.choose n s * k.descFactorial s * (k ^ 2 - k) ^ (n - s)) := by
            refine Finset.sum_congr rfl ?_
            intro s hs
            rw [Nat.mul_assoc]
