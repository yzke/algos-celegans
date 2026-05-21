# Edge sign audit (Phase 1.5+.1)

> Scope: verify the sign-assignment rule used by
> `src/algos/connectome.py` and `src/algos/graph/loader.py`. Identify
> known exceptions and flag uncertainties. **No code changes in this
> sub-phase** — findings live here and feed Phase 1.5+.4
> (`data_audit.md`) and the Phase 1.5 design.

Status of `src/algos/`: untouched by this audit.

---

## 1. Connectome data version

| field | value |
|---|---|
| Source workbook | `data/connectome/SI5_corrected.xlsx` |
| Origin | WormWiring, "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx" |
| Primary publication | Cook, Jarrell, Brittin, Bloniarz, Hagmann, Hobert et al. (2019). *Whole-animal connectomes of both Caenorhabditis elegans sexes.* **Nature** 571(7763): 63–71. doi:10.1038/s41586-019-1352-7 |
| Sex | Hermaphrodite (sheets `hermaphrodite chemical`, `hermaphrodite gap jn symmetric`) |
| Neuron count | 302 (300 chemical-emitting rows + CANL/R) |
| Chemical directed edges (nonzero raw) | **3,709** |
| Total chemical EM serial-section weight | 20,965 |
| Gap-junction unordered pairs | **1,091** (2,182 directed entries after mirroring) |
| Gap-junction symmetry | exact in the corrected sheet (we re-enforce `maximum(W, W.T)` defensively) |
| Self-gap entries on the sheet | 14, zeroed at load (cancel algebraically in `gap_input`) |

The brief's expected range ("约 3700 化学 + 1100 电") matches exactly.
No version drift detected.

---

## 2. Current sign-assignment rule

Implemented in `src/algos/neurotransmitters.py` + applied in
`src/algos/connectome.py::_from_xlsx`:

```
sign(pre) = -1  if pre ∈ GABAERGIC else +1
W_chem[i, j] = sign(j) × W_chem_raw[i, j]
```

`GABAERGIC` is a frozen set of 26 neurons:

```
AVL, DVB, RIS
RMED, RMEV, RMEL, RMER
DD01..DD06
VD01..VD13
```

Sources cited in `neurotransmitters.py`:

- McIntire SL, Jorgensen E, Kaplan J, Horvitz HR (1993). *The GABAergic
  nervous system of Caenorhabditis elegans.* **Nature** 364: 337–341.
  — the original RMEs / AVL / DVB / RIS / D-class identification.
- Schuske K, Beg AA, Jorgensen EM (2004). *The GABA nervous system in
  C. elegans.* **Trends Neurosci** 27(7): 407–414.
- Gendrel M, Atlas EG, Hobert O (2016). *A cellular and regulatory map
  of the GABAergic nervous system of C. elegans.* **eLife** 5: e17686.
  — extended canonical list (consistent with the 26 above).

The Gendrel 2016 paper also identifies ALA, AVB, AVJ, RIB, SMD as
"reuse" GABA neurons (they express the GABA biosynthesis machinery for
specific contexts) but the *primary* secretion at the connectome's
chemical synapses is not GABA for those cells. Excluding them from
`GABAERGIC` matches the standard treatment in connectome dynamics
papers.

**Verdict on the current rule:** sound for the 26 canonical
GABAergic neurons. The remaining 276 neurons are uniformly assigned
sign +1, which is a coarse approximation — see §3 for cases where
that approximation is known to be wrong or incomplete.

---

## 3. Known exceptions / co-transmission

The "one neuron, one transmitter" framing baked into the current rule
breaks down for the following well-documented cases. None of these
requires a code change for Phase 1.0, but each is a possible source of
the residual "forward ↔ reversal command +0.51 correlation" anomaly
(PHASE1.0_REPORT.md §4) and they need to be recorded.

### 3.1 Glutamate + monoamine co-release

**RIM (RIML/R)** — releases **glutamate** AND **tyramine**.
- Pirri JK, McPherson AD, Donnelly JL, Francis MM, Alkema MJ (2009).
  *A tyramine-gated chloride channel coordinates distinct motor programs
  of a Caenorhabditis elegans escape response.* **Neuron** 62: 526–538.
- Alkema MJ, Hunter-Ensor M, Ringstad N, Horvitz HR (2005).
  *Tyramine functions independently of octopamine in the Caenorhabditis
  elegans nervous system.* **Neuron** 46: 247–260.

Our handling: RIM is treated as `default` (excitatory glutamate
synapses). The tyramine arm is missing entirely. Tyramine acts on
LGC-55 (a tyramine-gated chloride channel) on AVB and several head
motor neurons → **inhibits forward locomotion during escape**. This
is exactly the kind of mutual-exclusion mechanism Phase 1.0 is
missing.

### 3.2 Acetylcholine + serotonin co-release

**HSN (HSNL/R)** — releases **acetylcholine** AND **serotonin**.
- Desai C, Garriga G, McIntire SL, Horvitz HR (1988). *A genetic
  pathway for the development of the Caenorhabditis elegans HSN motor
  neurons.* **Nature** 336: 638–646.
- Sze JY, Victor M, Loer C, Shi Y, Ruvkun G (2000). *Food and
  metabolic signalling defects in a Caenorhabditis elegans
  serotonin-synthesis mutant.* **Nature** 403: 560–564.

Our handling: HSN is `default` (excitatory ACh). The 5-HT arm is
modeled separately in `algos.neural_v2.modulators` as a 5-HT producer.
**This is the correct split for our two-system architecture** —
chemical synapses are point-to-point ACh (default sign +1), the
serotonergic arm is modulator (parameter-level). No bug here.

**VC4/VC5** — release ACh AND 5-HT onto vulval muscles (Schafer 2005).
Same handling: ACh sign +1 in W_chem, 5-HT arm not separately modeled
(VCs are not in `SHT_SOURCE_NEURONS`; Phase 1.5+.3 should add them).

**NSM (NSML/R)** — primary 5-HT source. Also releases neuropeptides
(NLP-3, FLP-21). Our handling: `default` for chemical, separate 5-HT
in modulator bank. Correct split.

**ADF (ADFL/R)** — releases 5-HT (Loer & Kenyon 1993). Our handling:
`default` for chemical synapses, included in 5-HT modulator pool.
Correct.

### 3.3 Glutamate + neuropeptide co-release

**AIY, AIB, AVA, AVB, AVE, RIM** — all release peptides from the FLP
or NLP families in addition to their classical transmitter. The
classical transmitter (glutamate or ACh) drives the per-tick chemical
synapse; the peptide arm is slow / volume / modulator. Phase 1.0
captures only the classical arm.

Reference:
- Frooninckx L, Van Rompay L, Temmerman L, Van Sinay E, Beets I,
  Janssen T, Husson SJ, Schoofs L (2012). *Neuropeptide GPCRs in
  C. elegans.* **Front Endocrinol** 3: 167.

### 3.4 Excitatory-direction GABA

Most GABA in C. elegans is inhibitory through GABA-A-like UNC-49.
**Known exception**: GABA acting on **EXP-1** (the cation-permeable
GABA receptor on enteric muscle) is **excitatory**.

- Beg AA, Jorgensen EM (2003). *EXP-1 is an excitatory GABA-gated
  cation channel.* **Nat Neurosci** 6(11): 1145–1152.

This affects synapses from **AVL → enteric muscle** and **DVB →
enteric muscle**. Enteric muscle is not in the 302-neuron set so
these synapses do not appear in `W_chem` at all — but if Phase 2+
adds muscle cells to the graph, AVL and DVB will need a *per-target*
sign (not the per-source sign we use today).

**No correction needed in Phase 1.0**: both AVL and DVB are GABA
sources with their inhibitory effect on neuron targets (the only
targets in our graph). The exception only matters once muscle is in
scope.

### 3.5 Mono-aminergic neurons without classical chemical synapses

Some monoaminergic cells **do not form classical chemical synapses
with discrete release sites in the EM data, only volume release**. The
Cook 2019 chemical sheet still lists their EM-identifiable swellings
as synapses. Affected neurons:

- **CEPDL/R, CEPVL/R, ADEL/R, PDEL/R** (dopamine; Sulston, Dew &
  Brenner 1975).
- **RIH** (some 5-HT contacts may be volume; literature less clear).

Our handling: these are scored `default` sign +1 in `W_chem`. The
dopamine modulator pool is **not yet implemented** (Phase 1.5+.3 will
document). If implemented, the *chemical* arm should be kept (it's
the EM-identifiable contacts) and a separate dopamine modulator added
on top, exactly the same split used for 5-HT.

### 3.6 Summary of the 26 GABA neurons — are any an outlier?

Cross-checked each against Gendrel 2016 (Figures 2 + 3):

| neuron(s) | confirmed GABAergic | notes |
|---|---|---|
| AVL | ✓ | additionally co-releases acetylcholine (Saheki & Bargmann 2009) — same caveat as HSN (ACh arm not modeled, fine). |
| DVB | ✓ | confirmed GABA. |
| RIS | ✓ | confirmed GABA. |
| RMED, RMEV, RMEL, RMER | ✓ | all four RMEs. |
| DD01–DD06 | ✓ | confirmed GABAergic motor neurons. |
| VD01–VD13 | ✓ | confirmed GABAergic motor neurons. |

No false positives in our GABA list. **Verdict: no correction needed.**

---

## 4. Gap-junction directionality

### 4.1 Current treatment

`src/algos/graph/loader.py` mirrors every electrical edge into two
directed entries with identical weight and `sign=+1`. The gap matrix
is enforced symmetric (`maximum(W, W.T)`) at load. The runtime's gap
input is the Laplacian form `W_gap @ V - V * sum(W_gap, axis=1)`,
which is symmetric in the strong sense — `V_i ↔ V_j` produces equal
and opposite currents.

### 4.2 Biology

Gap junctions in C. elegans are formed by **innexins** (inx-1 through
inx-22, plus unc-7, unc-9, eat-5, che-7). Many innexin combinations
form **rectifying** (asymmetric) junctions — current flows
preferentially in one direction. Documented examples:

- **AVA → A-class motor neurons** (backward locomotion) — rectifying
  via UNC-7 and UNC-9 heteromers. Forward current AVA→VA dominates.
  - Starich TA, Xu J, Skerrett IM, Nicholson BJ, Shaw JE (2009).
    *Interactions between innexins UNC-7 and UNC-9 mediate
    electrical synapse specificity in the Caenorhabditis elegans
    locomotory nervous system.* **Neural Dev** 4: 16.
  - Liu P, Chen B, Wang ZW (2017). *Postsynaptic current bursts
    instruct action potential firing at a graded synapse.* **Nat
    Commun** 8: 1376.

- **AVB → B-class motor neurons** (forward locomotion) — also
  rectifying.
  - Kawano T, Po MD, Gao S, Leung G, Ryu WS, Zhen M (2011). *An
    imbalancing act: gap junctions reduce the backward motor circuit
    activity to bias C. elegans for forward locomotion.* **Neuron**
    72: 572–586.

- **Several pharyngeal junctions** (M-class to muscles) — known
  rectifying.

### 4.3 Impact on Phase 1.0

The Cook 2019 SI 5 sheet **does not encode rectification**: it stores
only contact count, treated symmetric. There is no dataset within
the standard connectome distributions that records per-edge
rectification polarity. To add rectification properly, each
edge would need an annotation drawn from the innexin literature paper
by paper.

Our symmetric assumption is therefore a known approximation, not a
bug in the loader. **For Phase 1.0 it almost certainly contributes to
the forward ↔ reversal command +0.51 correlation**: in the real
worm, AVA and AVB are electrically coupled to *different* motor pools
through rectifying junctions, so a depolarization in AVA does not
fully flow back into AVB (and vice versa). In our symmetric model,
it does — and that drags AVA and AVB toward synchrony.

### 4.4 Recommendation

- **No Phase 1.0 code change** (consistent with the brief).
- **Phase 1.5 design** should consider per-edge `rectification` field
  on `Edge` for electrical type (default +1 = symmetric, value in
  [0, 1] gives forward bias). Concrete neurons to annotate first:
  AVA↔VA/DA, AVB↔VB/DB, RIM↔AVA, RMD intra-class.
- Track-and-defer in `QUESTIONS.md`.

---

## 5. Things this audit could NOT verify

Logged here and forwarded to `QUESTIONS.md`:

- **Q1**: Specific innexin composition of every electrical edge in
  Cook 2019 — required to know which gap junctions rectify in which
  direction. Likely 50+ edges have published rectification data
  scattered across separate papers; assembling a per-edge table is
  itself a literature-review project.
- **Q2**: Whether the 14 self-loop entries in the gap sheet that we
  zero at load (`src/algos/connectome.py`, "data artifacts") represent
  real intra-cellular gap junctions (some innexins are intracellular
  channels — UNC-9 forms hemichannels at the cell membrane), or
  scanning artifacts. The algebraic cancellation in `gap_input` makes
  it moot for dynamics, but the data interpretation is unclear.
- **Q3**: Are any of the "co-release neurons" in §3 producing tonic
  release vs. spike-triggered release? Tonic release implies a
  *background* effect not captured by spike-driven event-queue
  delivery.

---

## 6. Bottom line

| dimension | status |
|---|---|
| Connectome data version | ✓ correct (Cook 2019 corrected July 2020) |
| Edge count (chem / gap) | ✓ matches expected 3709 / 1091 |
| GABA neuron list (26) | ✓ correct, no false positives |
| Sign rule for non-GABA neurons | ✓ correct as approximation; known exceptions documented (§3) |
| Co-release handling | ✓ correct split for 5-HT modulator architecture; tyramine arm of RIM **missing** (§3.1) — relevant to forward↔reversal anomaly |
| Excitatory GABA (EXP-1) | ✓ irrelevant in Phase 1.0 (target is muscle, out of scope) |
| Gap junction rectification | ✗ known approximation; symmetric assumption may contribute to AVA/AVB co-firing |
| Self-gap entries (14, zeroed) | ✓ algebraically OK; data interpretation flagged in Q2 |

**No code changes warranted for Phase 1.0.** Two real findings for
Phase 1.5 design:
1. Add tyramine arm of RIM (would target AVB, MC, RMD via LGC-55 →
   chloride → suppress forward during escape).
2. Add per-edge rectification annotation for electrical edges
   (start with AVA↔A-class and AVB↔B-class).
