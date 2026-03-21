import lean.Basic

open Classical
open Polynomial

abbrev FreeColumn (k : ℕ) := Fin 3 ↪ Fin k

lemma freeColumnCard (k : ℕ) : Fintype.card (FreeColumn k) = k.descFactorial 3 := by
  rw [Fintype.card_embedding_eq]
  simp [Fintype.card_fin]

noncomputable def gPoly (k : ℕ) : Polynomial ℕ :=
  X + (1 + C (k - 1) * X) ^ 3

noncomputable def sCoeff (k m : ℕ) : ℕ :=
  ((gPoly k) ^ k).coeff m

noncomputable def T3Formula (k n : ℕ) : ℕ :=
  Finset.sum (Finset.range (Nat.min (3 * k) n + 1))
    (fun m => sCoeff k m * n.descFactorial m * (k.descFactorial 3) ^ (n - m))

noncomputable def weightedPoly {β : Type} [Fintype β] (w : β → ℕ) (r : ℕ) : Polynomial ℕ :=
  ∑ f : Fin r → β, X ^ ∑ i, w (f i)

lemma weightedPoly_zero {β : Type} [Fintype β] (w : β → ℕ) :
    weightedPoly w 0 = 1 := by
  simp [weightedPoly]

lemma weightedPoly_succ {β : Type} [Fintype β] (w : β → ℕ) (r : ℕ) :
    weightedPoly w (r + 1) = weightedPoly w r * (∑ b : β, X ^ w b) := by
  classical
  have hsplit :
      weightedPoly w (r + 1) = ∑ p : (Fin r → β) × β, X ^ (∑ i, w (p.1 i) + w p.2) := by
        unfold weightedPoly
        simpa [Fin.cons, Fin.tail, Fin.sum_univ_succ, add_comm, add_left_comm, add_assoc] using
          (Equiv.sum_comp ((Equiv.prodComm (Fin r → β) β).trans (Fin.consEquiv fun _ => β))
            (fun f : Fin (r + 1) → β => X ^ ∑ i, w (f i))).symm
  rw [hsplit]
  calc
    ∑ p : (Fin r → β) × β, X ^ (∑ i, w (p.1 i) + w p.2)
        = ∑ p : (Fin r → β) × β, (X ^ ∑ i, w (p.1 i)) * (X ^ w p.2) := by
            exact Fintype.sum_congr
              (fun p : (Fin r → β) × β => X ^ (∑ i, w (p.1 i) + w p.2))
              (fun p : (Fin r → β) × β => (X ^ ∑ i, w (p.1 i)) * (X ^ w p.2))
              (by
                intro p
                simp [pow_add])
    _ = ∑ t : Fin r → β, ∑ b : β, (X ^ ∑ i, w (t i)) * (X ^ w b) := by
            exact Fintype.sum_prod_type (fun p : (Fin r → β) × β => (X ^ ∑ i, w (p.1 i)) * (X ^ w p.2))
    _ = ∑ t : Fin r → β, (X ^ ∑ i, w (t i)) * (∑ b : β, X ^ w b) := by
            exact Fintype.sum_congr
              (fun t : Fin r → β => ∑ b : β, (X ^ ∑ i, w (t i)) * (X ^ w b))
              (fun t : Fin r → β => (X ^ ∑ i, w (t i)) * (∑ b : β, X ^ w b))
              (by
                intro t
                simpa using
                  (Finset.mul_sum
                    (s := Finset.univ)
                    (f := fun b : β => X ^ w b)
                    (a := X ^ ∑ i, w (t i))).symm)
    _ = weightedPoly w r * (∑ b : β, X ^ w b) := by
            rw [← Finset.sum_mul]
            rfl

lemma weightedPoly_eq_pow {β : Type} [Fintype β] (w : β → ℕ) (r : ℕ) :
    weightedPoly w r = (∑ b : β, X ^ w b) ^ r := by
  induction r with
  | zero =>
      simp [weightedPoly_zero]
  | succ r ihr =>
      rw [weightedPoly_succ, ihr, pow_succ, mul_comm]

lemma coeff_weightedPoly_eq_card {β : Type} [Fintype β] (w : β → ℕ) (r m : ℕ) :
    (weightedPoly w r).coeff m = Fintype.card {f : Fin r → β // ∑ i, w (f i) = m} := by
  classical
  unfold weightedPoly
  let coeffHom : Polynomial ℕ →+ ℕ :=
    { toFun := fun p => p.coeff m
      map_zero' := by simp
      map_add' := by
        intro p q
        simp }
  have hcoeff :
      (∑ f : Fin r → β, X ^ ∑ i, w (f i)).coeff m
        = ∑ f : Fin r → β, (((X : Polynomial ℕ) ^ ∑ i, w (f i)).coeff m) := by
          exact
            map_sum coeffHom
              (fun f : Fin r → β => (X ^ ∑ i, w (f i) : Polynomial ℕ))
              (Finset.univ : Finset (Fin r → β))
  rw [hcoeff]
  change (∑ f : Fin r → β, (((X : Polynomial ℕ) ^ ∑ i, w (f i)).coeff m)) =
    Fintype.card {f : Fin r → β // ∑ i, w (f i) = m}
  simp [Polynomial.coeff_X_pow, eq_comm]
  symm
  let s : Finset (Fin r → β) := {f ∈ (Finset.univ : Finset (Fin r → β)) | m = ∑ i, w (f i)}
  have hs :
      Fintype.card {f : Fin r → β | m = ∑ i, w (f i)} = s.card := by
    exact Fintype.card_ofFinset (p := {f : Fin r → β | m = ∑ i, w (f i)}) s (by
      intro f
      simp [s])
  have hs' :
      Fintype.card {f : Fin r → β // m = ∑ i, w (f i)} = s.card := by
    simpa using hs
  simpa [s] using hs'

def optionWeight {q : ℕ} : Option (Fin q) → ℕ
  | none => 0
  | some _ => 1

lemma optionPoly_eq (q : ℕ) :
    (∑ o : Option (Fin q), X ^ optionWeight o) = 1 + C q * X := by
  rw [Fintype.sum_option]
  simp [optionWeight, Fintype.card_fin]

abbrev PairPattern (k : ℕ) := Fin 3 → Option (Fin (k - 1))

def pairPatternSize {k : ℕ} (f : PairPattern k) : ℕ :=
  ∑ i, optionWeight (f i)

lemma pairPatternPoly_eq (k : ℕ) :
    weightedPoly (β := Option (Fin (k - 1))) optionWeight 3 = (1 + C (k - 1) * X) ^ 3 := by
  rw [weightedPoly_eq_pow, optionPoly_eq]

abbrev ColourPattern (k : ℕ) := Unit ⊕ PairPattern k

def colourPatternSize {k : ℕ} : ColourPattern k → ℕ
  | Sum.inl _ => 1
  | Sum.inr f => pairPatternSize f

lemma colourPatternPoly_eq_gPoly (k : ℕ) :
    (∑ p : ColourPattern k, X ^ colourPatternSize p) = gPoly k := by
  rw [Fintype.sum_sum_type]
  calc
    (∑ u : Unit, X ^ colourPatternSize (Sum.inl u)) + ∑ f : PairPattern k, X ^ colourPatternSize (Sum.inr f)
        = X + weightedPoly (β := Option (Fin (k - 1))) optionWeight 3 := by
            simp [colourPatternSize, pairPatternSize, weightedPoly]
    _ = gPoly k := by
            rw [pairPatternPoly_eq]
            simp [gPoly]

def sCount (k m : ℕ) : ℕ :=
  Fintype.card {f : Fin k → ColourPattern k // ∑ i, colourPatternSize (f i) = m}

lemma sCount_eq_sCoeff (k m : ℕ) : sCount k m = sCoeff k m := by
  unfold sCount sCoeff
  rw [← coeff_weightedPoly_eq_card (w := colourPatternSize) (r := k) (m := m),
    weightedPoly_eq_pow, colourPatternPoly_eq_gPoly]
