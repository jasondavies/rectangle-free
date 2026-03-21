import lean.Basic

open Classical

abbrev RowPair5 := Fin 10

def rowPair5 : RowPair5 → Fin 5 × Fin 5
  | 0 => (0, 1)
  | 1 => (0, 2)
  | 2 => (0, 3)
  | 3 => (0, 4)
  | 4 => (1, 2)
  | 5 => (1, 3)
  | 6 => (1, 4)
  | 7 => (2, 3)
  | 8 => (2, 4)
  | 9 => (3, 4)

abbrev Bucket5 := RowPair5 × Fin 4

abbrev Column5 := Fin 5 → Fin 4

lemma bucket5Card : Fintype.card Bucket5 = 40 := by
  simp [Bucket5, Fintype.card_prod, Fintype.card_fin]

def usesBucket5 (col : Column5) : Bucket5 → Prop
  | ⟨p, c⟩ =>
      let rs := rowPair5 p
      col rs.1 = c ∧ col rs.2 = c

noncomputable def bucketSet5 (col : Column5) : Finset Bucket5 :=
  Finset.univ.filter fun b => usesBucket5 col b

lemma rowPair5_ne (p : RowPair5) : (rowPair5 p).1 ≠ (rowPair5 p).2 := by
  fin_cases p <;> decide

def pairIndex5 : (r₁ r₂ : Fin 5) → r₁ ≠ r₂ → RowPair5
  | 0, 0, h => False.elim (h rfl)
  | 0, 1, _ => 0
  | 0, 2, _ => 1
  | 0, 3, _ => 2
  | 0, 4, _ => 3
  | 1, 0, _ => 0
  | 1, 1, h => False.elim (h rfl)
  | 1, 2, _ => 4
  | 1, 3, _ => 5
  | 1, 4, _ => 6
  | 2, 0, _ => 1
  | 2, 1, _ => 4
  | 2, 2, h => False.elim (h rfl)
  | 2, 3, _ => 7
  | 2, 4, _ => 8
  | 3, 0, _ => 2
  | 3, 1, _ => 5
  | 3, 2, _ => 7
  | 3, 3, h => False.elim (h rfl)
  | 3, 4, _ => 9
  | 4, 0, _ => 3
  | 4, 1, _ => 6
  | 4, 2, _ => 8
  | 4, 3, _ => 9
  | 4, 4, h => False.elim (h rfl)

def pairBucket5 (r₁ r₂ : Fin 5) (h : r₁ ≠ r₂) (c : Fin 4) : Bucket5 :=
  (pairIndex5 r₁ r₂ h, c)

lemma usesBucket5_pairBucket5_of_eq (col : Column5) (r₁ r₂ : Fin 5) (h : r₁ ≠ r₂)
    (c : Fin 4) (hr₁ : col r₁ = c) (hr₂ : col r₂ = c) :
    usesBucket5 col (pairBucket5 r₁ r₂ h c) := by
  fin_cases r₁ <;> fin_cases r₂ <;>
    simp [pairBucket5, pairIndex5, usesBucket5, rowPair5, hr₁, hr₂] at * <;>
    try cases (h rfl)

lemma pairBucket5_mem_bucketSet5_of_eq (col : Column5) (r₁ r₂ : Fin 5) (h : r₁ ≠ r₂)
    (c : Fin 4) (hr₁ : col r₁ = c) (hr₂ : col r₂ = c) :
    pairBucket5 r₁ r₂ h c ∈ bucketSet5 col := by
  have huse : usesBucket5 col (pairBucket5 r₁ r₂ h c) :=
    usesBucket5_pairBucket5_of_eq col r₁ r₂ h c hr₁ hr₂
  simpa [bucketSet5] using huse

def columnAt5 {n : ℕ} (A : Coloring 4 5 n) (c : Fin n) : Column5 :=
  fun r => A r c

lemma rectangleFree_disjoint5 {n : ℕ} {A : Coloring 4 5 n} (hA : RectangleFree A)
    {c₁ c₂ : Fin n} (hc : c₁ ≠ c₂) :
    Disjoint (bucketSet5 (columnAt5 A c₁)) (bucketSet5 (columnAt5 A c₂)) := by
  rw [Finset.disjoint_left]
  intro b hb₁ hb₂
  rcases b with ⟨p, c⟩
  have huse₁ : usesBucket5 (columnAt5 A c₁) (p, c) := by
    simpa [bucketSet5] using hb₁
  have huse₂ : usesBucket5 (columnAt5 A c₂) (p, c) := by
    simpa [bucketSet5] using hb₂
  let rs := rowPair5 p
  have hrs : rs.1 ≠ rs.2 := rowPair5_ne p
  have hrow : A rs.1 c₁ = A rs.1 c₂ := huse₁.1.trans huse₂.1.symm
  have hcol₁ : A rs.1 c₁ = A rs.2 c₁ := huse₁.1.trans huse₁.2.symm
  have hcol₂ : A rs.1 c₁ = A rs.2 c₂ := huse₁.1.trans huse₂.2.symm
  exact hA ⟨rs.1, rs.2, c₁, c₂, hrs, hc, hrow, hcol₁, hcol₂⟩

lemma column5_not_injective (col : Column5) : ¬ Function.Injective col := by
  intro hcol
  have hcard : Fintype.card (Fin 5) ≤ Fintype.card (Fin 4) :=
    Fintype.card_le_of_injective col hcol
  norm_num [Fintype.card_fin] at hcard

lemma bucketSet5_nonempty (col : Column5) : bucketSet5 col ≠ ∅ := by
  have hnotinj : ¬ Function.Injective col := column5_not_injective col
  rcases (by simpa [Function.Injective, and_comm, and_left_comm, and_assoc] using hnotinj :
      ∃ r₁ r₂, r₁ ≠ r₂ ∧ col r₁ = col r₂) with ⟨r₁, r₂, hr, hEq⟩
  rw [← Finset.nonempty_iff_ne_empty]
  refine ⟨pairBucket5 r₁ r₂ hr (col r₁), ?_⟩
  exact pairBucket5_mem_bucketSet5_of_eq col r₁ r₂ hr (col r₁) rfl hEq.symm

noncomputable def chooseBucket5 {n : ℕ} (A : Coloring 4 5 n) : Fin n → Bucket5 := fun c =>
  Classical.choose (Finset.nonempty_iff_ne_empty.mpr (bucketSet5_nonempty (columnAt5 A c)))

lemma chooseBucket5_mem {n : ℕ} (A : Coloring 4 5 n) (c : Fin n) :
    chooseBucket5 A c ∈ bucketSet5 (columnAt5 A c) := by
  exact Classical.choose_spec (Finset.nonempty_iff_ne_empty.mpr (bucketSet5_nonempty (columnAt5 A c)))

lemma chooseBucket5_injective {n : ℕ} {A : Coloring 4 5 n} (hA : RectangleFree A) :
    Function.Injective (chooseBucket5 A) := by
  intro c₁ c₂ hEq
  by_contra hc
  have hdisj : Disjoint (bucketSet5 (columnAt5 A c₁)) (bucketSet5 (columnAt5 A c₂)) :=
    rectangleFree_disjoint5 hA hc
  have hmem₁ : chooseBucket5 A c₁ ∈ bucketSet5 (columnAt5 A c₁) := chooseBucket5_mem A c₁
  have hmem₂ : chooseBucket5 A c₁ ∈ bucketSet5 (columnAt5 A c₂) := by
    simpa [hEq] using chooseBucket5_mem A c₂
  have hmemInter :
      chooseBucket5 A c₁ ∈ bucketSet5 (columnAt5 A c₁) ∩ bucketSet5 (columnAt5 A c₂) := by
    simp [hmem₁, hmem₂]
  have hinter : bucketSet5 (columnAt5 A c₁) ∩ bucketSet5 (columnAt5 A c₂) = ∅ := by
    exact Finset.disjoint_iff_inter_eq_empty.mp hdisj
  simp [hinter] at hmemInter

def pairwiseDisjointBucketSeq5 {n : ℕ} (f : Fin n → Column5) : Prop :=
  Pairwise fun i j => Disjoint (bucketSet5 (f i)) (bucketSet5 (f j))

abbrev CompatibleColumnSeq5 (n : ℕ) :=
  {f : Fin n → Column5 // pairwiseDisjointBucketSeq5 f}

noncomputable def Ccoeff5 (n : ℕ) : ℕ := by
  classical
  exact Fintype.card (CompatibleColumnSeq5 n)

def buildColoring5 {n : ℕ} (f : Fin n → Column5) : Coloring 4 5 n :=
  fun r c => f c r

lemma rectangleFree_buildColoring5 {n : ℕ} {f : CompatibleColumnSeq5 n} :
    RectangleFree (buildColoring5 f.1) := by
  intro hrect
  rcases hrect with ⟨r₁, r₂, c₁, c₂, hr, hc, hrow, hcol₁, hcol₂⟩
  let x : Fin 4 := buildColoring5 f.1 r₁ c₁
  have hmem₁ : pairBucket5 r₁ r₂ hr x ∈ bucketSet5 (f.1 c₁) := by
    refine pairBucket5_mem_bucketSet5_of_eq (f.1 c₁) r₁ r₂ hr x ?_ ?_
    · rfl
    · simpa [x, buildColoring5] using hcol₁.symm
  have hmem₂ : pairBucket5 r₁ r₂ hr x ∈ bucketSet5 (f.1 c₂) := by
    refine pairBucket5_mem_bucketSet5_of_eq (f.1 c₂) r₁ r₂ hr x ?_ ?_
    · simpa [x, buildColoring5] using hrow.symm
    · simpa [x, buildColoring5] using hcol₂.symm
  have hdisj : Disjoint (bucketSet5 (f.1 c₁)) (bucketSet5 (f.1 c₂)) := f.2 hc
  have hmemInter : pairBucket5 r₁ r₂ hr x ∈ bucketSet5 (f.1 c₁) ∩ bucketSet5 (f.1 c₂) := by
    simp [hmem₁, hmem₂]
  have hinter : bucketSet5 (f.1 c₁) ∩ bucketSet5 (f.1 c₂) = ∅ := by
    exact Finset.disjoint_iff_inter_eq_empty.mp hdisj
  rw [hinter] at hmemInter
  simp at hmemInter

noncomputable def rectangleFreeEquiv5 (n : ℕ) :
    {A : Coloring 4 5 n // RectangleFree A} ≃ CompatibleColumnSeq5 n :=
  { toFun := fun A =>
      ⟨fun c => columnAt5 A.1 c, by
        intro c₁ c₂ hc
        simpa using rectangleFree_disjoint5 A.2 hc⟩
    invFun := fun f => ⟨buildColoring5 f.1, rectangleFree_buildColoring5⟩
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

theorem T_4_5xn (n : ℕ) : T 4 5 n = Ccoeff5 n := by
  simpa [T, Ccoeff5] using Fintype.card_congr (rectangleFreeEquiv5 n)

abbrev Mask5 := {S : Finset Bucket5 // S ≠ ∅}

def pairwiseDisjointMaskSeq5 {n : ℕ} (m : Fin n → Mask5) : Prop :=
  Pairwise fun i j => Disjoint (m i).1 (m j).1

abbrev CompatibleMaskSeq5 (n : ℕ) :=
  {m : Fin n → Mask5 // pairwiseDisjointMaskSeq5 m}

noncomputable def maskWeight5 (S : Mask5) : ℕ := by
  classical
  exact Fintype.card {col : Column5 // bucketSet5 col = S.1}

abbrev maskChoice5 {n : ℕ} (m : CompatibleMaskSeq5 n) :=
  (i : Fin n) → {col : Column5 // bucketSet5 col = (m.1 i).1}

noncomputable def weightedCount5 (n : ℕ) : ℕ :=
  ∑ m : CompatibleMaskSeq5 n, ∏ i, maskWeight5 (m.1 i)

noncomputable def maskSeqOfColumns5 {n : ℕ} (f : CompatibleColumnSeq5 n) : CompatibleMaskSeq5 n :=
  ⟨fun i => ⟨bucketSet5 (f.1 i), bucketSet5_nonempty (f.1 i)⟩, by
    intro i j hij
    exact f.2 hij⟩

abbrev Fiber5 {n : ℕ} (f : CompatibleColumnSeq5 n → CompatibleMaskSeq5 n)
    (m : CompatibleMaskSeq5 n) :=
  {x : CompatibleColumnSeq5 n // f x = m}

instance fiberFintype5 {n : ℕ} (f : CompatibleColumnSeq5 n → CompatibleMaskSeq5 n)
    (m : CompatibleMaskSeq5 n) : Fintype (Fiber5 f m) :=
  Subtype.fintype fun x => f x = m

noncomputable def fiberEquivMaskChoice5 {n : ℕ} (m : CompatibleMaskSeq5 n) :
    Fiber5 maskSeqOfColumns5 m ≃ maskChoice5 m :=
  { toFun := fun f i =>
      ⟨f.1.1 i, by
        have hEq : (maskSeqOfColumns5 f.1).1 i = m.1 i := by
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

lemma maskChoiceCard5 {n : ℕ} (m : CompatibleMaskSeq5 n) :
    Fintype.card (maskChoice5 m) = ∏ i, maskWeight5 (m.1 i) := by
  classical
  simpa [maskChoice5, maskWeight5] using
    (Fintype.card_pi (α := fun i : Fin n => {col : Column5 // bucketSet5 col = (m.1 i).1}))

lemma cardSigmaFiber5 {n : ℕ} (f : CompatibleColumnSeq5 n → CompatibleMaskSeq5 n) :
    Fintype.card ((m : CompatibleMaskSeq5 n) × Fiber5 f m) =
      ∑ m : CompatibleMaskSeq5 n, Fintype.card (Fiber5 f m) := by
  rw [Fintype.card_sigma]

theorem Ccoeff5_eq_weightedCount5 (n : ℕ) : Ccoeff5 n = weightedCount5 n := by
  let f : CompatibleColumnSeq5 n → CompatibleMaskSeq5 n := maskSeqOfColumns5
  calc
    Ccoeff5 n
        = Fintype.card ((m : CompatibleMaskSeq5 n) × Fiber5 f m) := by
            simpa [Ccoeff5, f] using Fintype.card_congr (Equiv.sigmaFiberEquiv f).symm
    _ = ∑ m : CompatibleMaskSeq5 n, Fintype.card (Fiber5 f m) := by
            exact cardSigmaFiber5 f
    _ = ∑ m : CompatibleMaskSeq5 n, Fintype.card (maskChoice5 m) := by
            refine Finset.sum_congr rfl ?_
            intro m hm
            simpa [Fiber5, f] using Fintype.card_congr (fiberEquivMaskChoice5 m)
    _ = weightedCount5 n := by
            rw [weightedCount5]
            refine Finset.sum_congr rfl ?_
            intro m hm
            exact maskChoiceCard5 m

theorem T_4_5xn_weighted (n : ℕ) : T 4 5 n = weightedCount5 n := by
  rw [T_4_5xn, Ccoeff5_eq_weightedCount5]

lemma rectangleFree_card_le_40 {n : ℕ} {A : Coloring 4 5 n} (hA : RectangleFree A) : n ≤ 40 := by
  have hcard : Fintype.card (Fin n) ≤ Fintype.card Bucket5 :=
    Fintype.card_le_of_injective (chooseBucket5 A) (chooseBucket5_injective hA)
  simpa [bucket5Card, Fintype.card_fin] using hcard

theorem T_4_5xn_eq_zero_of_gt_40 (n : ℕ) (hn : 40 < n) : T 4 5 n = 0 := by
  letI : IsEmpty {A : Coloring 4 5 n // RectangleFree A} := by
    refine ⟨?_⟩
    intro A
    exact (Nat.not_lt_of_ge (rectangleFree_card_le_40 A.2)) hn
  simp [T]
