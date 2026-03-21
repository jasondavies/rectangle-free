import lean.Basic

open Classical

abbrev RowPair6 := Fin 15

def rowPair6 : RowPair6 → Fin 6 × Fin 6
  | 0 => (0, 1)
  | 1 => (0, 2)
  | 2 => (0, 3)
  | 3 => (0, 4)
  | 4 => (0, 5)
  | 5 => (1, 2)
  | 6 => (1, 3)
  | 7 => (1, 4)
  | 8 => (1, 5)
  | 9 => (2, 3)
  | 10 => (2, 4)
  | 11 => (2, 5)
  | 12 => (3, 4)
  | 13 => (3, 5)
  | 14 => (4, 5)

abbrev Bucket6 := RowPair6 × Fin 4

abbrev Column6 := Fin 6 → Fin 4

lemma bucket6Card : Fintype.card Bucket6 = 60 := by
  simp [Bucket6, Fintype.card_prod, Fintype.card_fin]

def usesBucket6 (col : Column6) : Bucket6 → Prop
  | ⟨p, c⟩ =>
      let rs := rowPair6 p
      col rs.1 = c ∧ col rs.2 = c

noncomputable def bucketSet6 (col : Column6) : Finset Bucket6 :=
  Finset.univ.filter fun b => usesBucket6 col b

lemma rowPair6_ne (p : RowPair6) : (rowPair6 p).1 ≠ (rowPair6 p).2 := by
  fin_cases p <;> decide

def pairIndex6 : (r₁ r₂ : Fin 6) → r₁ ≠ r₂ → RowPair6
  | 0, 0, h => False.elim (h rfl)
  | 0, 1, _ => 0
  | 0, 2, _ => 1
  | 0, 3, _ => 2
  | 0, 4, _ => 3
  | 0, 5, _ => 4
  | 1, 0, _ => 0
  | 1, 1, h => False.elim (h rfl)
  | 1, 2, _ => 5
  | 1, 3, _ => 6
  | 1, 4, _ => 7
  | 1, 5, _ => 8
  | 2, 0, _ => 1
  | 2, 1, _ => 5
  | 2, 2, h => False.elim (h rfl)
  | 2, 3, _ => 9
  | 2, 4, _ => 10
  | 2, 5, _ => 11
  | 3, 0, _ => 2
  | 3, 1, _ => 6
  | 3, 2, _ => 9
  | 3, 3, h => False.elim (h rfl)
  | 3, 4, _ => 12
  | 3, 5, _ => 13
  | 4, 0, _ => 3
  | 4, 1, _ => 7
  | 4, 2, _ => 10
  | 4, 3, _ => 12
  | 4, 4, h => False.elim (h rfl)
  | 4, 5, _ => 14
  | 5, 0, _ => 4
  | 5, 1, _ => 8
  | 5, 2, _ => 11
  | 5, 3, _ => 13
  | 5, 4, _ => 14
  | 5, 5, h => False.elim (h rfl)

def pairBucket6 (r₁ r₂ : Fin 6) (h : r₁ ≠ r₂) (c : Fin 4) : Bucket6 :=
  (pairIndex6 r₁ r₂ h, c)

lemma usesBucket6_pairBucket6_of_eq (col : Column6) (r₁ r₂ : Fin 6) (h : r₁ ≠ r₂)
    (c : Fin 4) (hr₁ : col r₁ = c) (hr₂ : col r₂ = c) :
    usesBucket6 col (pairBucket6 r₁ r₂ h c) := by
  fin_cases r₁ <;> fin_cases r₂ <;>
    simp [pairBucket6, pairIndex6, usesBucket6, rowPair6, hr₁, hr₂] at * <;>
    try cases (h rfl)

lemma pairBucket6_mem_bucketSet6_of_eq (col : Column6) (r₁ r₂ : Fin 6) (h : r₁ ≠ r₂)
    (c : Fin 4) (hr₁ : col r₁ = c) (hr₂ : col r₂ = c) :
    pairBucket6 r₁ r₂ h c ∈ bucketSet6 col := by
  have huse : usesBucket6 col (pairBucket6 r₁ r₂ h c) :=
    usesBucket6_pairBucket6_of_eq col r₁ r₂ h c hr₁ hr₂
  simpa [bucketSet6] using huse

def columnAt6 {n : ℕ} (A : Coloring 4 6 n) (c : Fin n) : Column6 :=
  fun r => A r c

lemma rectangleFree_disjoint6 {n : ℕ} {A : Coloring 4 6 n} (hA : RectangleFree A)
    {c₁ c₂ : Fin n} (hc : c₁ ≠ c₂) :
    Disjoint (bucketSet6 (columnAt6 A c₁)) (bucketSet6 (columnAt6 A c₂)) := by
  rw [Finset.disjoint_left]
  intro b hb₁ hb₂
  rcases b with ⟨p, c⟩
  have huse₁ : usesBucket6 (columnAt6 A c₁) (p, c) := by
    simpa [bucketSet6] using hb₁
  have huse₂ : usesBucket6 (columnAt6 A c₂) (p, c) := by
    simpa [bucketSet6] using hb₂
  let rs := rowPair6 p
  have hrs : rs.1 ≠ rs.2 := rowPair6_ne p
  have hrow : A rs.1 c₁ = A rs.1 c₂ := huse₁.1.trans huse₂.1.symm
  have hcol₁ : A rs.1 c₁ = A rs.2 c₁ := huse₁.1.trans huse₁.2.symm
  have hcol₂ : A rs.1 c₁ = A rs.2 c₂ := huse₁.1.trans huse₂.2.symm
  exact hA ⟨rs.1, rs.2, c₁, c₂, hrs, hc, hrow, hcol₁, hcol₂⟩

lemma column6_not_injective (col : Column6) : ¬ Function.Injective col := by
  intro hcol
  have hcard : Fintype.card (Fin 6) ≤ Fintype.card (Fin 4) :=
    Fintype.card_le_of_injective col hcol
  norm_num [Fintype.card_fin] at hcard

lemma bucketSet6_nonempty (col : Column6) : bucketSet6 col ≠ ∅ := by
  have hnotinj : ¬ Function.Injective col := column6_not_injective col
  rcases (by simpa [Function.Injective, and_comm, and_left_comm, and_assoc] using hnotinj :
      ∃ r₁ r₂, r₁ ≠ r₂ ∧ col r₁ = col r₂) with ⟨r₁, r₂, hr, hEq⟩
  rw [← Finset.nonempty_iff_ne_empty]
  refine ⟨pairBucket6 r₁ r₂ hr (col r₁), ?_⟩
  exact pairBucket6_mem_bucketSet6_of_eq col r₁ r₂ hr (col r₁) rfl hEq.symm

noncomputable def chooseBucket6 {n : ℕ} (A : Coloring 4 6 n) : Fin n → Bucket6 := fun c =>
  Classical.choose (Finset.nonempty_iff_ne_empty.mpr (bucketSet6_nonempty (columnAt6 A c)))

lemma chooseBucket6_mem {n : ℕ} (A : Coloring 4 6 n) (c : Fin n) :
    chooseBucket6 A c ∈ bucketSet6 (columnAt6 A c) := by
  exact Classical.choose_spec (Finset.nonempty_iff_ne_empty.mpr (bucketSet6_nonempty (columnAt6 A c)))

lemma chooseBucket6_injective {n : ℕ} {A : Coloring 4 6 n} (hA : RectangleFree A) :
    Function.Injective (chooseBucket6 A) := by
  intro c₁ c₂ hEq
  by_contra hc
  have hdisj : Disjoint (bucketSet6 (columnAt6 A c₁)) (bucketSet6 (columnAt6 A c₂)) :=
    rectangleFree_disjoint6 hA hc
  have hmem₁ : chooseBucket6 A c₁ ∈ bucketSet6 (columnAt6 A c₁) := chooseBucket6_mem A c₁
  have hmem₂ : chooseBucket6 A c₁ ∈ bucketSet6 (columnAt6 A c₂) := by
    simpa [hEq] using chooseBucket6_mem A c₂
  have hmemInter :
      chooseBucket6 A c₁ ∈ bucketSet6 (columnAt6 A c₁) ∩ bucketSet6 (columnAt6 A c₂) := by
    simp [hmem₁, hmem₂]
  have hinter : bucketSet6 (columnAt6 A c₁) ∩ bucketSet6 (columnAt6 A c₂) = ∅ := by
    exact Finset.disjoint_iff_inter_eq_empty.mp hdisj
  simp [hinter] at hmemInter

def pairwiseDisjointBucketSeq6 {n : ℕ} (f : Fin n → Column6) : Prop :=
  Pairwise fun i j => Disjoint (bucketSet6 (f i)) (bucketSet6 (f j))

abbrev CompatibleColumnSeq6 (n : ℕ) :=
  {f : Fin n → Column6 // pairwiseDisjointBucketSeq6 f}

noncomputable def Ccoeff6 (n : ℕ) : ℕ := by
  classical
  exact Fintype.card (CompatibleColumnSeq6 n)

def buildColoring6 {n : ℕ} (f : Fin n → Column6) : Coloring 4 6 n :=
  fun r c => f c r

lemma rectangleFree_buildColoring6 {n : ℕ} {f : CompatibleColumnSeq6 n} :
    RectangleFree (buildColoring6 f.1) := by
  intro hrect
  rcases hrect with ⟨r₁, r₂, c₁, c₂, hr, hc, hrow, hcol₁, hcol₂⟩
  let x : Fin 4 := buildColoring6 f.1 r₁ c₁
  have hmem₁ : pairBucket6 r₁ r₂ hr x ∈ bucketSet6 (f.1 c₁) := by
    refine pairBucket6_mem_bucketSet6_of_eq (f.1 c₁) r₁ r₂ hr x ?_ ?_
    · rfl
    · simpa [x, buildColoring6] using hcol₁.symm
  have hmem₂ : pairBucket6 r₁ r₂ hr x ∈ bucketSet6 (f.1 c₂) := by
    refine pairBucket6_mem_bucketSet6_of_eq (f.1 c₂) r₁ r₂ hr x ?_ ?_
    · simpa [x, buildColoring6] using hrow.symm
    · simpa [x, buildColoring6] using hcol₂.symm
  have hdisj : Disjoint (bucketSet6 (f.1 c₁)) (bucketSet6 (f.1 c₂)) := f.2 hc
  have hmemInter : pairBucket6 r₁ r₂ hr x ∈ bucketSet6 (f.1 c₁) ∩ bucketSet6 (f.1 c₂) := by
    simp [hmem₁, hmem₂]
  have hinter : bucketSet6 (f.1 c₁) ∩ bucketSet6 (f.1 c₂) = ∅ := by
    exact Finset.disjoint_iff_inter_eq_empty.mp hdisj
  rw [hinter] at hmemInter
  simp at hmemInter

noncomputable def rectangleFreeEquiv6 (n : ℕ) :
    {A : Coloring 4 6 n // RectangleFree A} ≃ CompatibleColumnSeq6 n :=
  { toFun := fun A =>
      ⟨fun c => columnAt6 A.1 c, by
        intro c₁ c₂ hc
        simpa using rectangleFree_disjoint6 A.2 hc⟩
    invFun := fun f => ⟨buildColoring6 f.1, rectangleFree_buildColoring6⟩
    left_inv := by
      intro A
      apply Subtype.ext
      funext r c
      rfl
    right_inv := by
      intro f
      apply Subtype.ext
      funext c
      funext r
      rfl }

theorem T_4_6xn (n : ℕ) : T 4 6 n = Ccoeff6 n := by
  simpa [T, Ccoeff6] using Fintype.card_congr (rectangleFreeEquiv6 n)

abbrev Mask6 := {S : Finset Bucket6 // S ≠ ∅}

def pairwiseDisjointMaskSeq6 {n : ℕ} (m : Fin n → Mask6) : Prop :=
  Pairwise fun i j => Disjoint (m i).1 (m j).1

abbrev CompatibleMaskSeq6 (n : ℕ) :=
  {m : Fin n → Mask6 // pairwiseDisjointMaskSeq6 m}

noncomputable def maskWeight6 (S : Mask6) : ℕ := by
  classical
  exact Fintype.card {col : Column6 // bucketSet6 col = S.1}

abbrev maskChoice6 {n : ℕ} (m : CompatibleMaskSeq6 n) :=
  (i : Fin n) → {col : Column6 // bucketSet6 col = (m.1 i).1}

noncomputable def weightedCount6 (n : ℕ) : ℕ :=
  ∑ m : CompatibleMaskSeq6 n, ∏ i, maskWeight6 (m.1 i)

noncomputable def maskSeqOfColumns6 {n : ℕ} (f : CompatibleColumnSeq6 n) : CompatibleMaskSeq6 n :=
  ⟨fun i => ⟨bucketSet6 (f.1 i), bucketSet6_nonempty (f.1 i)⟩, by
    intro i j hij
    exact f.2 hij⟩

abbrev Fiber6 {n : ℕ} (f : CompatibleColumnSeq6 n → CompatibleMaskSeq6 n)
    (m : CompatibleMaskSeq6 n) :=
  {x : CompatibleColumnSeq6 n // f x = m}

instance fiberFintype6 {n : ℕ} (f : CompatibleColumnSeq6 n → CompatibleMaskSeq6 n)
    (m : CompatibleMaskSeq6 n) : Fintype (Fiber6 f m) :=
  Subtype.fintype fun x => f x = m

noncomputable def fiberEquivMaskChoice6 {n : ℕ} (m : CompatibleMaskSeq6 n) :
    Fiber6 maskSeqOfColumns6 m ≃ maskChoice6 m :=
  { toFun := fun f i =>
      ⟨f.1.1 i, by
        have hEq : (maskSeqOfColumns6 f.1).1 i = m.1 i := by
          exact congrFun (congrArg Subtype.val f.2) i
        exact congrArg Subtype.val hEq⟩
    invFun := fun g =>
      ⟨⟨fun i => (g i).1, by
          intro i j hij
          simpa [(g i).2, (g j).2] using m.2 hij⟩, by
        apply Subtype.ext
        funext i
        apply Subtype.ext
        exact (g i).2⟩
    left_inv := by
      intro f
      apply Subtype.ext
      apply Subtype.ext
      funext i
      rfl
    right_inv := by
      intro g
      funext i
      apply Subtype.ext
      rfl }

lemma maskChoiceCard6 {n : ℕ} (m : CompatibleMaskSeq6 n) :
    Fintype.card (maskChoice6 m) = ∏ i, maskWeight6 (m.1 i) := by
  classical
  simpa [maskChoice6, maskWeight6] using
    (Fintype.card_pi (α := fun i : Fin n => {col : Column6 // bucketSet6 col = (m.1 i).1}))

lemma cardSigmaFiber6 {n : ℕ} (f : CompatibleColumnSeq6 n → CompatibleMaskSeq6 n) :
    Fintype.card ((m : CompatibleMaskSeq6 n) × Fiber6 f m) =
      ∑ m : CompatibleMaskSeq6 n, Fintype.card (Fiber6 f m) := by
  rw [Fintype.card_sigma]

theorem Ccoeff6_eq_weightedCount6 (n : ℕ) : Ccoeff6 n = weightedCount6 n := by
  let f : CompatibleColumnSeq6 n → CompatibleMaskSeq6 n := maskSeqOfColumns6
  calc
    Ccoeff6 n
        = Fintype.card ((m : CompatibleMaskSeq6 n) × Fiber6 f m) := by
            simpa [Ccoeff6, f] using Fintype.card_congr (Equiv.sigmaFiberEquiv f).symm
    _ = ∑ m : CompatibleMaskSeq6 n, Fintype.card (Fiber6 f m) := by
            exact cardSigmaFiber6 f
    _ = ∑ m : CompatibleMaskSeq6 n, Fintype.card (maskChoice6 m) := by
            refine Finset.sum_congr rfl ?_
            intro m hm
            simpa [Fiber6, f] using Fintype.card_congr (fiberEquivMaskChoice6 m)
    _ = weightedCount6 n := by
            rw [weightedCount6]
            refine Finset.sum_congr rfl ?_
            intro m hm
            exact maskChoiceCard6 m

theorem T_4_6xn_weighted (n : ℕ) : T 4 6 n = weightedCount6 n := by
  rw [T_4_6xn, Ccoeff6_eq_weightedCount6]

lemma rectangleFree_card_le_60 {n : ℕ} {A : Coloring 4 6 n} (hA : RectangleFree A) : n ≤ 60 := by
  have hcard : Fintype.card (Fin n) ≤ Fintype.card Bucket6 :=
    Fintype.card_le_of_injective (chooseBucket6 A) (chooseBucket6_injective hA)
  simpa [bucket6Card, Fintype.card_fin] using hcard

theorem T_4_6xn_eq_zero_of_gt_60 (n : ℕ) (hn : 60 < n) : T 4 6 n = 0 := by
  letI : IsEmpty {A : Coloring 4 6 n // RectangleFree A} := by
    refine ⟨?_⟩
    intro A
    exact (Nat.not_lt_of_ge (rectangleFree_card_le_60 A.2)) hn
  simp [T]
