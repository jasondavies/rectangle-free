import lean.Basic

open Classical

abbrev RowPair4 := Fin 6

def rowPair4 : RowPair4 → Fin 4 × Fin 4
  | 0 => (0, 1)
  | 1 => (0, 2)
  | 2 => (0, 3)
  | 3 => (1, 2)
  | 4 => (1, 3)
  | 5 => (2, 3)

abbrev Bucket4 (k : ℕ) := RowPair4 × Fin k

abbrev Column4 (k : ℕ) := Fin 4 → Fin k

abbrev FreeColumn4 (k : ℕ) := Fin 4 ↪ Fin k

lemma freeColumn4Card (k : ℕ) : Fintype.card (FreeColumn4 k) = k.descFactorial 4 := by
  rw [Fintype.card_embedding_eq]
  simp [Fintype.card_fin]

lemma bucket4Card (k : ℕ) : Fintype.card (Bucket4 k) = 6 * k := by
  simp [Bucket4, Fintype.card_prod, Fintype.card_fin]

def usesBucket4 {k : ℕ} (col : Column4 k) : Bucket4 k → Prop
  | ⟨p, c⟩ =>
      let rs := rowPair4 p
      col rs.1 = c ∧ col rs.2 = c

noncomputable def bucketSet4 {k : ℕ} (col : Column4 k) : Finset (Bucket4 k) :=
  Finset.univ.filter fun b => usesBucket4 col b

abbrev NonfreeColumn4 (k : ℕ) := {col : Column4 k // bucketSet4 col ≠ ∅}

noncomputable def pairwiseDisjointBucketSeq4 {k t : ℕ} (f : Fin t → NonfreeColumn4 k) : Prop :=
  Pairwise fun i j => Disjoint (bucketSet4 (f i).1) (bucketSet4 (f j).1)

abbrev CompatibleNonfreeSeq4 (k t : ℕ) :=
  {f : Fin t → NonfreeColumn4 k // pairwiseDisjointBucketSeq4 f}

noncomputable def Bcoeff4 (k t : ℕ) : ℕ := by
  classical
  exact Fintype.card {f : Fin t → NonfreeColumn4 k // pairwiseDisjointBucketSeq4 f}

noncomputable def T4Formula (k n : ℕ) : ℕ :=
  Finset.sum (Finset.range (Nat.min (6 * k) n + 1))
    (fun t => Nat.choose n t * (k.descFactorial 4) ^ (n - t) * Bcoeff4 k t)

lemma rowPair4_ne (p : RowPair4) : (rowPair4 p).1 ≠ (rowPair4 p).2 := by
  fin_cases p <;> decide

def pairIndex4 : (r₁ r₂ : Fin 4) → r₁ ≠ r₂ → RowPair4
  | 0, 0, h => False.elim (h rfl)
  | 0, 1, _ => 0
  | 0, 2, _ => 1
  | 0, 3, _ => 2
  | 1, 0, _ => 0
  | 1, 1, h => False.elim (h rfl)
  | 1, 2, _ => 3
  | 1, 3, _ => 4
  | 2, 0, _ => 1
  | 2, 1, _ => 3
  | 2, 2, h => False.elim (h rfl)
  | 2, 3, _ => 5
  | 3, 0, _ => 2
  | 3, 1, _ => 4
  | 3, 2, _ => 5
  | 3, 3, h => False.elim (h rfl)

def pairBucket4 {k : ℕ} (r₁ r₂ : Fin 4) (h : r₁ ≠ r₂) (c : Fin k) : Bucket4 k :=
  (pairIndex4 r₁ r₂ h, c)

lemma usesBucket4_pairBucket4_of_eq {k : ℕ} (col : Column4 k) (r₁ r₂ : Fin 4) (h : r₁ ≠ r₂)
    (c : Fin k) (hr₁ : col r₁ = c) (hr₂ : col r₂ = c) :
    usesBucket4 col (pairBucket4 r₁ r₂ h c) := by
  fin_cases r₁ <;> fin_cases r₂ <;>
    simp [pairBucket4, pairIndex4, usesBucket4, rowPair4, hr₁, hr₂] at * <;>
    try cases (h rfl)

lemma pairBucket4_mem_bucketSet4_of_eq {k : ℕ} (col : Column4 k) (r₁ r₂ : Fin 4) (h : r₁ ≠ r₂)
    (c : Fin k) (hr₁ : col r₁ = c) (hr₂ : col r₂ = c) :
    pairBucket4 r₁ r₂ h c ∈ bucketSet4 col := by
  have huse : usesBucket4 col (pairBucket4 r₁ r₂ h c) :=
    usesBucket4_pairBucket4_of_eq col r₁ r₂ h c hr₁ hr₂
  simpa [bucketSet4] using huse

def columnAt4 {k n : ℕ} (A : Coloring k 4 n) (c : Fin n) : Column4 k :=
  fun r => A r c

noncomputable def nonfreeSet4 {k n : ℕ} (A : Coloring k 4 n) : Finset (Fin n) :=
  Finset.univ.filter fun c => bucketSet4 (columnAt4 A c) ≠ ∅

lemma mem_nonfreeSet4_iff {k n : ℕ} (A : Coloring k 4 n) (c : Fin n) :
    c ∈ nonfreeSet4 A ↔ bucketSet4 (columnAt4 A c) ≠ ∅ := by
  simp [nonfreeSet4]

lemma injective_of_bucketSet4_eq_empty {k : ℕ} {col : Column4 k} (hcol : bucketSet4 col = ∅) :
    Function.Injective col := by
  intro r₁ r₂ hEq
  by_contra hne
  have hmem : pairBucket4 r₁ r₂ hne (col r₁) ∈ bucketSet4 col := by
    exact pairBucket4_mem_bucketSet4_of_eq col r₁ r₂ hne (col r₁) rfl hEq.symm
  rw [hcol] at hmem
  simp at hmem

lemma bucketSet4_embedding_eq_empty {k : ℕ} (e : FreeColumn4 k) : bucketSet4 e = ∅ := by
  ext b
  constructor
  · intro hb
    rcases b with ⟨p, c⟩
    have huse : usesBucket4 (fun r => e r) (p, c) := by
      simpa [bucketSet4] using hb
    exact False.elim (rowPair4_ne p (e.injective (huse.1.trans huse.2.symm)))
  · intro hb
    cases hb

def pairwiseDisjointBucketFamily4 {k n : ℕ} {S : Finset (Fin n)}
    (f : ↥S → NonfreeColumn4 k) : Prop :=
  Pairwise fun i j => Disjoint (bucketSet4 (f i).1) (bucketSet4 (f j).1)

abbrev CompatibleNonfreeFamily4 (k n : ℕ) (S : Finset (Fin n)) :=
  {f : ↥S → NonfreeColumn4 k // pairwiseDisjointBucketFamily4 f}

abbrev RectFiber4 (k n : ℕ) (S : Finset (Fin n)) :=
  {A : Coloring k 4 n // RectangleFree A ∧ nonfreeSet4 A = S}

abbrev RectData4 (k n : ℕ) (S : Finset (Fin n)) :=
  (↥(Sᶜ) → FreeColumn4 k) × CompatibleNonfreeFamily4 k n S

lemma rectangleFree_disjoint4 {k n : ℕ} {A : Coloring k 4 n} (hA : RectangleFree A)
    {c₁ c₂ : Fin n} (hc : c₁ ≠ c₂) :
    Disjoint (bucketSet4 (columnAt4 A c₁)) (bucketSet4 (columnAt4 A c₂)) := by
  rw [Finset.disjoint_left]
  intro b hb₁ hb₂
  rcases b with ⟨p, c⟩
  have huse₁ : usesBucket4 (columnAt4 A c₁) (p, c) := by
    simpa [bucketSet4] using hb₁
  have huse₂ : usesBucket4 (columnAt4 A c₂) (p, c) := by
    simpa [bucketSet4] using hb₂
  let rs := rowPair4 p
  have hrs : rs.1 ≠ rs.2 := rowPair4_ne p
  have hrow : A rs.1 c₁ = A rs.1 c₂ := huse₁.1.trans huse₂.1.symm
  have hcol₁ : A rs.1 c₁ = A rs.2 c₁ := huse₁.1.trans huse₁.2.symm
  have hcol₂ : A rs.1 c₁ = A rs.2 c₂ := huse₁.1.trans huse₂.2.symm
  exact hA ⟨rs.1, rs.2, c₁, c₂, hrs, hc, hrow, hcol₁, hcol₂⟩

def nonfreeFunction4 {k n : ℕ} (A : Coloring k 4 n) (S : Finset (Fin n))
    (hS : nonfreeSet4 A = S) : ↥S → NonfreeColumn4 k := fun c =>
  ⟨columnAt4 A c, by
    have hmem : (c : Fin n) ∈ nonfreeSet4 A := by
      simpa [hS] using c.2
    exact (mem_nonfreeSet4_iff A c).1 hmem⟩

def freeFunction4 {k n : ℕ} (A : Coloring k 4 n) (S : Finset (Fin n))
    (hS : nonfreeSet4 A = S) : ↥(Sᶜ) → FreeColumn4 k := fun c =>
  let col := columnAt4 A c
  let hempty : bucketSet4 col = ∅ := by
    have hnot : (c : Fin n) ∉ nonfreeSet4 A := by
      simpa [hS] using (Finset.mem_compl.mp c.2)
    have : ¬ bucketSet4 col ≠ ∅ := by
      intro hne
      exact hnot ((mem_nonfreeSet4_iff A c).2 hne)
    simpa using this
  { toFun := col
    inj' := injective_of_bucketSet4_eq_empty hempty }

def nonfreeFamily4 {k n : ℕ} (A : Coloring k 4 n) (hA : RectangleFree A) (S : Finset (Fin n))
    (hS : nonfreeSet4 A = S) : CompatibleNonfreeFamily4 k n S :=
  ⟨nonfreeFunction4 A S hS, by
    intro c₁ c₂ hc
    have hval : (c₁ : Fin n) ≠ c₂ := by
      intro hEq
      apply hc
      exact Subtype.ext hEq
    simpa [nonfreeFunction4] using rectangleFree_disjoint4 hA hval⟩

def buildColoring4 {k n : ℕ} (S : Finset (Fin n)) (d : RectData4 k n S) : Coloring k 4 n :=
  fun r c =>
    if h : c ∈ S then
      (d.2.1 ⟨c, h⟩).1 r
    else
      d.1 ⟨c, by simpa [Finset.mem_compl] using h⟩ r

lemma columnAt4_buildColoring4_mem {k n : ℕ} (S : Finset (Fin n)) (d : RectData4 k n S)
    {c : Fin n} (h : c ∈ S) :
    columnAt4 (buildColoring4 S d) c = (d.2.1 ⟨c, h⟩).1 := by
  funext r
  simp [columnAt4, buildColoring4, h]

lemma columnAt4_buildColoring4_not_mem {k n : ℕ} (S : Finset (Fin n)) (d : RectData4 k n S)
    {c : Fin n} (h : c ∉ S) :
    columnAt4 (buildColoring4 S d) c = d.1 ⟨c, by simpa [Finset.mem_compl] using h⟩ := by
  funext r
  simp [columnAt4, buildColoring4, h]

lemma nonfreeSet4_buildColoring4 {k n : ℕ} (S : Finset (Fin n)) (d : RectData4 k n S) :
    nonfreeSet4 (buildColoring4 S d) = S := by
  ext c
  by_cases h : c ∈ S
  · have hne : bucketSet4 (columnAt4 (buildColoring4 S d) c) ≠ ∅ := by
      simpa [columnAt4_buildColoring4_mem S d h] using (d.2.1 ⟨c, h⟩).2
    simpa [h] using (mem_nonfreeSet4_iff _ _).2 hne
  · have hempty : bucketSet4 (columnAt4 (buildColoring4 S d) c) = ∅ := by
      simpa [columnAt4_buildColoring4_not_mem S d h] using
        bucketSet4_embedding_eq_empty (d.1 ⟨c, by simpa [Finset.mem_compl] using h⟩)
    have hnot : c ∉ nonfreeSet4 (buildColoring4 S d) := by
      intro hc
      exact by simpa [hempty] using (mem_nonfreeSet4_iff _ _).1 hc
    simpa [h] using hnot

lemma rectangleFree_buildColoring4 {k n : ℕ} (S : Finset (Fin n)) (d : RectData4 k n S) :
    RectangleFree (buildColoring4 S d) := by
  intro hrect
  rcases hrect with ⟨r₁, r₂, c₁, c₂, hr, hc, hrow, hcol₁, hcol₂⟩
  have hc₁S : c₁ ∈ S := by
    by_contra hc₁S
    have hEq : buildColoring4 S d r₁ c₁ = buildColoring4 S d r₂ c₁ := hcol₁
    have hcol :
        d.1 ⟨c₁, by simpa [Finset.mem_compl] using hc₁S⟩ r₁
          = d.1 ⟨c₁, by simpa [Finset.mem_compl] using hc₁S⟩ r₂ := by
      simpa [buildColoring4, hc₁S] using hEq
    exact hr ((d.1 ⟨c₁, by simpa [Finset.mem_compl] using hc₁S⟩).injective hcol)
  have hc₂eq : buildColoring4 S d r₁ c₂ = buildColoring4 S d r₂ c₂ := by
    calc
      buildColoring4 S d r₁ c₂ = buildColoring4 S d r₁ c₁ := hrow.symm
      _ = buildColoring4 S d r₂ c₂ := hcol₂
  have hc₂S : c₂ ∈ S := by
    by_contra hc₂S
    have hcol :
        d.1 ⟨c₂, by simpa [Finset.mem_compl] using hc₂S⟩ r₁
          = d.1 ⟨c₂, by simpa [Finset.mem_compl] using hc₂S⟩ r₂ := by
      simpa [buildColoring4, hc₂S] using hc₂eq
    exact hr ((d.1 ⟨c₂, by simpa [Finset.mem_compl] using hc₂S⟩).injective hcol)
  let x : Fin k := buildColoring4 S d r₁ c₁
  have hmem₁ : pairBucket4 r₁ r₂ hr x ∈ bucketSet4 ((d.2.1 ⟨c₁, hc₁S⟩).1) := by
    refine pairBucket4_mem_bucketSet4_of_eq ((d.2.1 ⟨c₁, hc₁S⟩).1) r₁ r₂ hr x ?_ ?_
    · simp [x, buildColoring4, hc₁S]
    · simpa [x, buildColoring4, hc₁S] using hcol₁.symm
  have hmem₂ : pairBucket4 r₁ r₂ hr x ∈ bucketSet4 ((d.2.1 ⟨c₂, hc₂S⟩).1) := by
    refine pairBucket4_mem_bucketSet4_of_eq ((d.2.1 ⟨c₂, hc₂S⟩).1) r₁ r₂ hr x ?_ ?_
    · simpa [x, buildColoring4, hc₂S] using hrow.symm
    · simpa [x, buildColoring4, hc₂S] using hcol₂.symm
  have hdisj :
      Disjoint (bucketSet4 ((d.2.1 ⟨c₁, hc₁S⟩).1)) (bucketSet4 ((d.2.1 ⟨c₂, hc₂S⟩).1)) :=
    d.2.2 (fun hEq => hc (by simpa using congrArg Subtype.val hEq))
  have hmemInter :
      pairBucket4 r₁ r₂ hr x ∈
        bucketSet4 ((d.2.1 ⟨c₁, hc₁S⟩).1) ∩ bucketSet4 ((d.2.1 ⟨c₂, hc₂S⟩).1) := by
    simp [hmem₁, hmem₂]
  have hinter :
      bucketSet4 ((d.2.1 ⟨c₁, hc₁S⟩).1) ∩ bucketSet4 ((d.2.1 ⟨c₂, hc₂S⟩).1) = ∅ := by
    exact (Finset.disjoint_iff_inter_eq_empty.mp hdisj)
  rw [hinter] at hmemInter
  simp at hmemInter

def rectFiberEquiv4 {k n : ℕ} (S : Finset (Fin n)) : RectFiber4 k n S ≃ RectData4 k n S :=
  { toFun := fun A => ⟨freeFunction4 A.1 S A.2.2, nonfreeFamily4 A.1 A.2.1 S A.2.2⟩
    invFun := fun d => ⟨buildColoring4 S d, rectangleFree_buildColoring4 S d, nonfreeSet4_buildColoring4 S d⟩
    left_inv := by
      intro A
      apply Subtype.ext
      funext r c
      by_cases h : c ∈ S
      · have hcol :
            columnAt4
                (buildColoring4 S
                  ⟨freeFunction4 A.1 S A.2.2, nonfreeFamily4 A.1 A.2.1 S A.2.2⟩) c
              =
            columnAt4 A.1 c := by
          simpa [nonfreeFunction4] using
            columnAt4_buildColoring4_mem S
              ⟨freeFunction4 A.1 S A.2.2, nonfreeFamily4 A.1 A.2.1 S A.2.2⟩ h
        simpa [columnAt4] using congrFun hcol r
      · have hcol :
            columnAt4
                (buildColoring4 S
                  ⟨freeFunction4 A.1 S A.2.2, nonfreeFamily4 A.1 A.2.1 S A.2.2⟩) c
              =
            columnAt4 A.1 c := by
          simpa [freeFunction4] using
            columnAt4_buildColoring4_not_mem S
              ⟨freeFunction4 A.1 S A.2.2, nonfreeFamily4 A.1 A.2.1 S A.2.2⟩ h
        simpa [columnAt4] using congrFun hcol r
    right_inv := by
      rintro ⟨ffree, fnonfree⟩
      apply Prod.ext
      · funext c
        have hnot : (c : Fin n) ∉ S := Finset.mem_compl.mp c.2
        have hcol :
            columnAt4 (buildColoring4 S (ffree, fnonfree)) c = ffree c := by
          simpa using columnAt4_buildColoring4_not_mem S (ffree, fnonfree) (c := c) hnot
        apply DFunLike.ext
        intro r
        exact congrFun hcol r
      · apply Subtype.ext
        funext c
        apply Subtype.ext
        have hcol :
            columnAt4 (buildColoring4 S (ffree, fnonfree)) c = (fnonfree.1 c).1 := by
          simpa using columnAt4_buildColoring4_mem S (ffree, fnonfree) (c := c) c.2
        funext r
        exact congrFun hcol r }

noncomputable def compatibleFamilyEquiv4 {k n : ℕ} (S : Finset (Fin n)) :
    CompatibleNonfreeFamily4 k n S ≃ CompatibleNonfreeSeq4 k S.card := by
  classical
  let e : ↥S ≃ Fin S.card := Fintype.equivFinOfCardEq (by simpa using Fintype.card_coe S)
  refine
    { toFun := fun f =>
        ⟨fun i => f.1 (e.symm i), by
          intro i j hij
          exact f.2 (fun hEq => hij (by simpa using congrArg e hEq))⟩
      invFun := fun f =>
        ⟨fun s => f.1 (e s), by
          intro i j hij
          exact f.2 (fun hEq => hij (by simpa using congrArg e.symm hEq))⟩
      left_inv := ?_
      right_inv := ?_ }
  · intro f
    apply Subtype.ext
    funext s
    simp [e]
  · intro f
    apply Subtype.ext
    funext i
    simp [e]

lemma rectData4Card {k n : ℕ} (S : Finset (Fin n)) :
    Fintype.card (RectData4 k n S) = (k.descFactorial 4) ^ (n - S.card) * Bcoeff4 k S.card := by
  have hFree : Fintype.card (↥(Sᶜ) → FreeColumn4 k) = (k.descFactorial 4) ^ (n - S.card) := by
    rw [Fintype.card_fun, freeColumn4Card]
    simp [Fintype.card_coe, Fintype.card_fin]
  have hNonfree : Fintype.card (CompatibleNonfreeFamily4 k n S) = Bcoeff4 k S.card := by
    simpa [Bcoeff4] using Fintype.card_congr (compatibleFamilyEquiv4 (k := k) (n := n) S)
  calc
    Fintype.card (RectData4 k n S)
        = Fintype.card (↥(Sᶜ) → FreeColumn4 k) * Fintype.card (CompatibleNonfreeFamily4 k n S) := by
            simp [RectData4, Fintype.card_prod]
    _ = (k.descFactorial 4) ^ (n - S.card) * Bcoeff4 k S.card := by rw [hFree, hNonfree]

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

lemma sum_rectData4_by_card (k n : ℕ) :
    (∑ S : Finset (Fin n), (k.descFactorial 4) ^ (n - S.card) * Bcoeff4 k S.card)
      =
    Finset.sum (Finset.range (n + 1))
      (fun s => Nat.choose n s * ((k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)) := by
  calc
    (∑ S : Finset (Fin n), (k.descFactorial 4) ^ (n - S.card) * Bcoeff4 k S.card)
        = Finset.sum
            ((Finset.range (n + 1)).biUnion
              (fun s => Finset.powersetCard s (Finset.univ : Finset (Fin n))))
            (fun S => (k.descFactorial 4) ^ (n - S.card) * Bcoeff4 k S.card) := by
              rw [powersetCard_biUnion_range]
    _ = Finset.sum (Finset.range (n + 1))
          (fun s =>
            Finset.sum (Finset.powersetCard s (Finset.univ : Finset (Fin n)))
              (fun S => (k.descFactorial 4) ^ (n - S.card) * Bcoeff4 k S.card)) := by
              exact Finset.sum_biUnion (powersetCard_pairwiseDisjoint n)
    _ = Finset.sum (Finset.range (n + 1))
          (fun s =>
            Finset.sum (Finset.powersetCard s (Finset.univ : Finset (Fin n)))
              (fun _S => (k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)) := by
              refine Finset.sum_congr rfl ?_
              intro s hs
              refine Finset.sum_congr rfl ?_
              intro S hS
              simp [(Finset.mem_powersetCard.1 hS).2]
    _ = Finset.sum (Finset.range (n + 1))
          (fun s => Nat.choose n s * ((k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)) := by
              refine Finset.sum_congr rfl ?_
              intro s hs
              have hconst :
                  Finset.sum (Finset.powersetCard s (Finset.univ : Finset (Fin n)))
                    (fun _S => (k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)
                    =
                  (Finset.powersetCard s (Finset.univ : Finset (Fin n))).card *
                    ((k.descFactorial 4) ^ (n - s) * Bcoeff4 k s) := by
                      exact
                        Finset.sum_const_nat
                          (s := Finset.powersetCard s (Finset.univ : Finset (Fin n)))
                          (m := (k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)
                          (fun _ _ => rfl)
              rw [hconst, Finset.card_powersetCard]
              simp [Finset.card_univ, Fintype.card_fin]

theorem T_4xn_raw (k n : ℕ) :
    T k 4 n =
      Finset.sum (Finset.range (n + 1))
        (fun s => Nat.choose n s * ((k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)) := by
  let f : {A : Coloring k 4 n // RectangleFree A} → Finset (Fin n) := fun A => nonfreeSet4 A.1
  calc
    T k 4 n
        = Fintype.card ((S : Finset (Fin n)) × {A : {A : Coloring k 4 n // RectangleFree A} // f A = S}) := by
            simpa [T, f] using
              Fintype.card_congr (Equiv.sigmaFiberEquiv f).symm
    _ = ∑ S : Finset (Fin n),
          Fintype.card {A : {A : Coloring k 4 n // RectangleFree A} // f A = S} := by
            rw [Fintype.card_sigma]
    _ = ∑ S : Finset (Fin n), Fintype.card (RectFiber4 k n S) := by
            refine Finset.sum_congr rfl ?_
            intro S hS
            simpa [f] using Fintype.card_congr
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
    _ = ∑ S : Finset (Fin n), Fintype.card (RectData4 k n S) := by
            refine Finset.sum_congr rfl ?_
            intro S hS
            simpa using Fintype.card_congr (rectFiberEquiv4 (k := k) (n := n) S)
    _ = ∑ S : Finset (Fin n), (k.descFactorial 4) ^ (n - S.card) * Bcoeff4 k S.card := by
            refine Finset.sum_congr rfl ?_
            intro S hS
            exact rectData4Card (k := k) (n := n) S
    _ = Finset.sum (Finset.range (n + 1))
          (fun s => Nat.choose n s * ((k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)) := by
            exact sum_rectData4_by_card k n

lemma compatibleNonfreeSeq4_isEmpty_of_lt {k t : ℕ} (ht : 6 * k < t) :
    IsEmpty (CompatibleNonfreeSeq4 k t) := by
  classical
  refine ⟨?_⟩
  intro f
  let chooseBucket : Fin t → Bucket4 k := fun i =>
    Classical.choose (Finset.nonempty_iff_ne_empty.mpr (f.1 i).2)
  have hchoose : ∀ i, chooseBucket i ∈ bucketSet4 (f.1 i).1 := by
    intro i
    exact Classical.choose_spec (Finset.nonempty_iff_ne_empty.mpr (f.1 i).2)
  have hinj : Function.Injective chooseBucket := by
    intro i j hij
    by_contra hne
    have hdisj : Disjoint (bucketSet4 (f.1 i).1) (bucketSet4 (f.1 j).1) := f.2 hne
    have hmemi : chooseBucket i ∈ bucketSet4 (f.1 i).1 := hchoose i
    have hmemj : chooseBucket i ∈ bucketSet4 (f.1 j).1 := by
      simpa [hij] using hchoose j
    have hmemInter :
        chooseBucket i ∈ bucketSet4 (f.1 i).1 ∩ bucketSet4 (f.1 j).1 := by
      simp [hmemi, hmemj]
    have hinter : bucketSet4 (f.1 i).1 ∩ bucketSet4 (f.1 j).1 = ∅ := by
      exact Finset.disjoint_iff_inter_eq_empty.mp hdisj
    simp [hinter] at hmemInter
  have hcard : Fintype.card (Fin t) ≤ Fintype.card (Bucket4 k) :=
    Fintype.card_le_of_injective chooseBucket hinj
  have hcard' : t ≤ 6 * k := by
    simpa [bucket4Card, Fintype.card_fin] using hcard
  exact Nat.not_lt.mpr hcard' ht

lemma Bcoeff4_eq_zero_of_lt {k t : ℕ} (ht : 6 * k < t) : Bcoeff4 k t = 0 := by
  classical
  letI := compatibleNonfreeSeq4_isEmpty_of_lt (k := k) (t := t) ht
  simp [Bcoeff4]

theorem T_4xn (k n : ℕ) : T k 4 n = T4Formula k n := by
  rw [T_4xn_raw, T4Formula]
  by_cases h : n ≤ 6 * k
  · simp [Nat.min_eq_right h, Nat.mul_assoc]
  · have hkn : 6 * k ≤ n := Nat.le_of_lt (Nat.lt_of_not_ge h)
    let f : ℕ → ℕ := fun s => Nat.choose n s * ((k.descFactorial 4) ^ (n - s) * Bcoeff4 k s)
    have hsubset : Finset.range (6 * k + 1) ⊆ Finset.range (n + 1) := by
      intro s hs
      refine Finset.mem_range.2 ?_
      exact lt_of_lt_of_le (Finset.mem_range.1 hs) (Nat.succ_le_succ hkn)
    have hzero : ∀ s ∈ Finset.range (n + 1), s ∉ Finset.range (6 * k + 1) → f s = 0 := by
      intro s hs hs'
      have hs'lt : ¬ s < 6 * k + 1 := by
        simpa using hs'
      have hsbig : 6 * k < s := Nat.succ_le_iff.mp (Nat.le_of_not_lt hs'lt)
      simp [f, Bcoeff4_eq_zero_of_lt hsbig]
    have hsum :
        Finset.sum (Finset.range (6 * k + 1)) f =
          Finset.sum (Finset.range (n + 1)) f :=
      Finset.sum_subset hsubset hzero
    simpa [Nat.min_eq_left hkn, f, Nat.mul_assoc] using hsum.symm
