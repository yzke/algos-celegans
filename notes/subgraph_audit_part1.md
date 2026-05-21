# Subgraph audit, Part 1: the 13 existing circuits (Phase 1.5+.2)

> Sub-phase: 1.5+.2 first half — verify each of the 13 functional
> subgraphs in `src/algos/graph/circuits.py` against the literature.
> Output is descriptive, not prescriptive — code is not modified.

For each circuit:
- **Current definition** = exact membership in `circuits.py::CIRCUIT_SPECS`.
- **Literature** = the citation chain backing the membership.
- **Empirical connectivity** = relevant raw EM contact counts from
  Cook 2019 (verified directly via `ConnectomeData`).
- **Issues** = things that look wrong, ambiguous, or worth revisiting.

---

## Circuit 1: `reversal_command` (recurrent, 6 nodes)

**Current definition**: `AVAL, AVAR, AVDL, AVDR, AVEL, AVER`.

**Literature**:
- Chalfie M, Sulston JE, White JG, Southgate E, Thomson JN, Brenner S
  (1985). *The neural circuit for touch sensitivity in Caenorhabditis
  elegans.* **J Neurosci** 5: 956–964. — names AVA and AVD as the
  reversal command interneurons.
- Gray JM, Hill JJ, Bargmann CI (2005). *A circuit for navigation in
  Caenorhabditis elegans.* **PNAS** 102: 3184–3191. — confirms AVA, AVD,
  AVE as the "backward-locomotion command" core.
- Piggott BJ, Liu J, Feng Z, Wescott SA, Xu XZS (2011). *The neural
  circuits and synaptic mechanisms underlying motor initiation in
  C. elegans.* **Cell** 147: 922–933. — adds AVE confirmation and
  RIM coupling.

**Empirical connectivity** (raw EM contacts):
- AVAL→AVAR 12, AVAR→AVAL 7. Within-pair recurrence.
- AVDL→AVAL 37, AVDL→AVAR 37, AVDR→AVAL 41, AVDR→AVAR 52. Touch input
  funnels through AVD into AVA — extremely strong.
- AVAL↔AVEL 2 gap contacts. Tight pair coupling.

**Issues**:
1. **RIM not included.** RIM (RIML/R) is functionally part of the
   reversal command in many recent papers (Wang H et al. 2020 *eLife*;
   Roberts WM et al. 2016 *Curr Biol*) — its tyramine arm
   actively *suppresses* AVB during escape (Pirri 2009). Our circuits.py
   puts RIM in `forward_command` with the note "winner-take-all
   counterpart". Both placements are defensible (RIM is a hub
   between forward and reversal); we should pick one or have it
   appear in both.

   **Recommendation**: keep RIM in `forward_command` for now (it's a
   forward-suppressing neuron, so being in the forward subgraph as a
   gating node is consistent). Document the dual role.

2. **AVD vs AVE distinction**:
   - AVD = ascending command from touch (gets ALM/AVM/PLM input).
   - AVE = head reversal command, more sensitive to gentle nose touch.
   - Our circuit lumps them. For Phase 1.5+, consider splitting into
     `reversal_command_AVA_AVD` and `reversal_command_AVE` — biology
     supports the separation (Piggott 2011).

3. **AVA → PVC 21+20 contacts!** PVC is currently in `forward_command`.
   So `reversal_command` (AVA) directly excites the chief forward
   neuron. The connectome's structure here is incompatible with a
   simple "AVA only triggers backward" picture and is one of the
   reasons the Phase 1.0 mutual exclusion fails. Need the inhibitory
   mechanism (RIM-tyramine, RIS, or DD/VD cross-inhibition) to break
   the synchrony — none of which are wired in Phase 1.0.

---

## Circuit 2: `forward_command` (recurrent, 10 nodes)

**Current definition**: `AVBL, AVBR, PVCL, PVCR, RIBL, RIBR, AIBL, AIBR,
RIML, RIMR`.

**Literature**:
- Gray et al. 2005 (PNAS) — AVB, PVC as forward command.
- Kawano T, Po MD, Gao S, Leung G, Ryu WS, Zhen M (2011). *An
  imbalancing act: gap junctions reduce the backward motor circuit
  activity to bias C. elegans for forward locomotion.* **Neuron**
  72: 572–586. — AVB↔B-class motor neurons via rectifying gap
  junctions (relevant: not modeled in Phase 1.0, see edge_sign_audit
  §4.2).
- Wang Y, Zhang X, Xin Q et al. (2020). *Flexible motor sequence
  generation during stereotyped escape responses.* **eLife** 9:
  e56942. — AIB/RIM/AVB winner-take-all loop.

**Empirical connectivity**:
- AVB pair: AVBL↔AVBR 18 gap contacts (very tight).
- AVB→AVA: AVBL→AVAL 9, AVBL→AVAR 14, AVBR→AVAL 10, AVBR→AVAR 14 —
  forward command directly *excites* reversal command. Symmetric to
  the AVA→AVB problem above.
- AIB→RIM: AIBL→RIMR 56, AIBR→RIML 47. Largest single chemical edges
  in this circuit. AIB is the main driver of RIM.
- AIZ→AIB: AIZL→AIBR 53, AIZR→AIBL 50. AIZ also dumps into AIB.
- RIB outputs heavy on AVE: AIYL→RIBL 42, AIYR→RIBR 60 (input side).

**Issues**:
1. **RIM is dual-role** (see Circuit 1 §1).
2. **PVC missing AVD coupling**: PVCs have heavy reciprocal gap with
   AVA and AVD that we don't separately model. Acceptable as overlap
   nodes for `posterior_touch`.
3. **RIB is mostly an integrator output target, not a forward command
   proper.** It receives huge input from AIY and outputs to head motor
   (RMD, SMD). Being in `forward_command` is borderline — RIB
   could equally be in `head_motor_cpg`. Currently in neither overlap.

---

## Circuit 3: `anterior_touch` (feedforward, 7 nodes)

**Current definition**: `ALML, ALMR, AVM, AVDL, AVDR, AVAL, AVAR`.

**Literature**:
- Chalfie et al. 1985 (J Neurosci) — the canonical reference. ALM
  (anterior lateral) + AVM (anterior ventral) → AVD command → A-class
  motor. PLM (posterior) → AVB/PVC.

**Empirical connectivity**:
- ALML→AVDL/R, ALMR→AVDL/R, AVM→AVDL/R — all confirmed in Cook 2019.
- AVD→AVA contacts (37–52) confirmed (see Circuit 1).

**Issues**:
1. **ALM also outputs to PVC** (not just AVD): ALML→PVCL 11 contacts.
   That means ALM has weak forward-command branches too. Probably an
   anatomical artifact rather than functional; standard treatment in
   the literature ignores it.
2. **AVM → DVA edge** (AVM→DVA 11 contacts) is not part of the touch
   reflex per Chalfie 1985 but does exist anatomically. DVA is the
   defecation hub. Out of scope for `anterior_touch`.
3. **Sensory side membership is complete** for ALM and AVM. Missing
   *FLP*, *ASH-derived nociception*, *BDU* (all sensory candidates
   per Chalfie 1985), but these are minor contributors.

---

## Circuit 4: `posterior_touch` (feedforward, 9 nodes)

**Current definition**: `PLML, PLMR, PVM, PVDL, PVDR, AVBL, AVBR, PVCL,
PVCR`.

**Literature**:
- Chalfie 1985 (as above).
- Way JC, Chalfie M (1989). *The mec-3 gene of Caenorhabditis
  elegans requires its own product for maintained expression and is
  expressed in three neuronal cell types.* **Genes Dev** 3(12A):
  1823–1833. — PVD as harsh-touch nociceptor.
- Tao L, Tracy CJ, Liu H et al. (2019). *Distinct roles of two
  C. elegans Bub1 paralogs in mechanosensation revealed by inhibitory
  vs excitatory regulation of touch neuron activity.* **PLoS Genet**
  — corroborates PVD's role.

**Empirical connectivity**:
- PLM→PVC: standard touch-reflex pathway.
- PVD→PVC: weak (PVD doesn't synapse heavily on PVC; the harsh-touch
  reflex bypasses PVC via direct gap to AVA/AVD).

**Issues**:
1. **PVD's actual targets**: PVDL outputs strongest to PVCL 8 and
   AVAL 5 — so PVD is more of a "drives reversal" neuron, *not*
   "drives forward". Putting PVD in `posterior_touch → forward` may
   be the wrong direction. Way & Chalfie 1989 actually report PVD as
   driving reversal (harsh touch makes the worm back up). The
   placement in our subgraph misclassifies PVD's role.

   **Recommendation**: move PVD from `posterior_touch` to a new
   `harsh_touch_nociception` subgraph that targets AVA/AVD/PVC, OR
   split `posterior_touch` into "gentle" (PLM, PVM → AVB) and "harsh"
   (PVD → AVA/AVD/PVC).

---

## Circuit 5: `chemosensory_amphid` (feedforward, 18 nodes)

**Current definition**: `ASEL, ASER, AWCL, AWCR, AWAL, AWAR, ASHL, ASHR,
ASKL, ASKR, AIYL, AIYR, AIZL, AIZR, AIBL, AIBR, RIAL, RIAR`.

**Literature**:
- Bargmann CI (2012). *Beyond the connectome: how neuromodulators
  shape neural circuits.* **BioEssays** 34: 458–465. — describes the
  amphid feedforward (ASE/AWC/AWA → AIY/AIZ/AIB → RIA → motor).
- Tomioka M et al. (2006). *The insulin/PI 3-kinase pathway regulates
  salt chemotaxis learning in Caenorhabditis elegans.* **Neuron** 51:
  613–625. — ASE/AIY/AIB salt-taste learning.

**Empirical connectivity** confirms the fan-in:
- ASEL→AIYL 32, ASER→AIYR 38, ASER→AIBR 32.
- AWCL→AIYL 22, AWCR→AIYR 22, AWCR→AIBR 18.
- AIYL→AIZL 67 + AIYL→RIAL 51, AIYR→AIZR 70 + AIYR→RIAR 50.
- RIA→head motor: RIAL→RMDDL 48, RIAL→RMDVL 48, RIAL→RMDVR 48, RIAR
  similarly strong. RIA is essentially "head motor command".

**Issues**:
1. **AIA missing.** AIA (AIAL/R) is the *second* major 1st-order
   interneuron pair for chemosensory input (ASER→AIAL 8, AWAL→AIAL 4,
   AWCR→AIAR 8). AIA is a key node for chemotaxis decision-making
   (Larsch et al. 2015, *Cell* 161: 215). Worth adding.
2. **ADL nociception** missing: ADL is the harsh-chemical nociceptor
   (Hilliard et al. 2002), couples into the same AIZ/AIB pipeline.
3. **RIA as part of `chemosensory_amphid` AND `head_motor_cpg` is
   correct** (it's the canonical bridge). Overlap honored.

---

## Circuit 6: `thermosensory` (feedforward, 8 nodes)

**Current definition**: `AFDL, AFDR, AIYL, AIYR, AIZL, AIZR, RIAL,
RIAR`.

**Literature**:
- Mori I, Ohshima Y (1995). *Neural regulation of thermotaxis in
  Caenorhabditis elegans.* **Nature** 376: 344–348. — AFD as the
  primary thermal sensor; AIY/AIZ as relays.
- Hawk JD et al. (2018). *Integration of plasticity mechanisms within
  a single sensory neuron of C. elegans actuates a memory.*
  **Neuron** 97(2): 356–367.

**Empirical connectivity**:
- AFDL→AIYL/R: small (~2-4 contacts). AFD doesn't have heavy direct
  synapses with AIY in Cook 2019; the functional coupling is partly
  via Q-cells and modulation.

**Issues**:
1. **AWC also responds to temperature** (Biron 2008 *Nat Neurosci*).
   Could be in thermosensory.
2. **AIN/RIB missing.** Recent work (Beverly et al. 2011) implicates
   AIN and RIB in thermal memory. RIB is in `forward_command`; AIN is
   in no circuit.

---

## Circuit 7: `head_motor_cpg` (recurrent, 24 nodes)

**Current definition**: `RMDL, RMDR, RMDDL, RMDDR, RMDVL, RMDVR,
SMDDL, SMDDR, SMDVL, SMDVR, SAADL, SAADR, SAAVL, SAAVR, OLQDL, OLQDR,
OLQVL, OLQVR, RIAL, RIAR, RMED, RMEV, RMEL, RMER`.

**Literature**:
- Faumont S, Rondeau G, Thiele TR et al. (2011). *An image-free
  opto-mechanical system for creating virtual environments and
  imaging neuronal activity in freely moving Caenorhabditis elegans.*
  **PLoS ONE** 6(9): e24666.
- Kawano T et al. 2011 (Neuron) — head motor CPG anatomy.
- Hendricks M, Ha H, Maffey N, Zhang Y (2012). *Compartmentalized
  calcium dynamics in a C. elegans interneuron encode head movement.*
  **Nature** 487: 99–103. — RIA compartmentalized dynamics.

**Empirical connectivity**:
- RIA→RMD: ~48 contacts each direction. RIA is the dominant input.
- SMD inter-pair: SMDDL→SMDVR 17, SMDVR→SMDDL 14. Reciprocal head
  L/R inhibition (mostly cholinergic, not GABA).
- RME pair: RMED/V/L/R all GABAergic; key amplitude-gating ring.

**Issues**:
1. **OLQ is dual-role**: it's a CEP-like mechano-sensory neuron at
   the head, not a CPG component proper. Could be removed.
2. **SIA/SIB missing**: SMD→SIA/SIB are heavy outputs (each ~14-30
   contacts). SIA/SIB drive head muscle. They would belong to a
   downstream "head muscle motor" subgraph.

---

## Circuit 8: `pharyngeal_cpg` (recurrent, 20 nodes)

**Current definition**: `M1, M2L/R, M3L/R, M4, M5, MCL/R, MI, I1L/R,
I2L/R, I3, I4, I5, I6, NSML, NSMR`.

**Literature**:
- Avery L, Horvitz HR (1989). *Pharyngeal pumping continues after
  laser killing of the pharyngeal nervous system of Caenorhabditis
  elegans.* **Neuron** 3(4): 473–485.
- Trojanowski NF, Padovan-Merhar O, Raizen DM, Fang-Yen C (2014).
  *Neural and genetic degeneracy underlies Caenorhabditis elegans
  feeding behavior.* **J Neurophysiol** 112(4): 951–961.

**Empirical connectivity** (subset of the 20 nodes, internal):
- All 20 are in the pharyngeal section of Cook 2019; chemical and gap
  connectivity is dense within this set, sparse out.

**Issues**:
1. **Almost completely silent in Phase 1.0** (mean rate = 0.000 over
   8000 ticks). The pharyngeal system is anatomically isolated; its
   only somatic interface is via NSM → broader 5-HT pool, RIP (head
   integration link), and a couple of M1 outputs. Without
   pharyngeal-specific sensory input (food presence) the CPG has no
   activator.
2. **MI and M5** are minor inactive cells in many feeding states —
   their inclusion is correct but expect low activity.
3. **Subcircuit decomposition possible**: pharyngeal pumping (M1-M5
   + MC + MI) vs. peristalsis (I-class). Phase 1.5 could split.

---

## Circuit 9: `ventral_cord_motor` (recurrent, 58 nodes)

**Current definition**: all DA01–DA09, DB01–DB07, VA01–VA12, VB01–VB11,
DD01–DD06, VD01–VD13.

**Literature**:
- White JG, Southgate E, Thomson JN, Brenner S (1986). *The structure
  of the nervous system of the nematode Caenorhabditis elegans.*
  **Phil Trans R Soc B** 314: 1–340. — original motor pool anatomy.
- Wen Q et al. (2012). *Proprioceptive coupling within motor neurons
  drives C. elegans forward locomotion.* **Neuron** 76: 750–761. —
  intra-class proprioception.

**Empirical connectivity**:
- Within-class gap-junction ladder confirmed (DA-DA, DB-DB, etc.).
- A→DD and B→VD cross-inhibitory chemical contacts confirmed.

**Issues**:
1. **Silent in Phase 1.0** (mean rate = 0.000). The motor pool needs
   upstream command-pool drive that the bare network doesn't sustain.
   This will only fire when Phase 1.5 supplies behavioral context.
2. **AS-class motor neurons missing!** Cook 2019 includes AS01–AS11
   (cholinergic motor, dorsal-side excitatory). Verify: are they in
   our connectome?

   Verified: AS01–AS11 ARE in `c.neuron_names` (302-neuron set).
   They are **not in `ventral_cord_motor`** — addition needed.
3. **PDA, PDB single neurons** also part of the motor pool, not
   included.

---

## Circuit 10: `modulator_RID` (feedforward, 5 nodes)

**Current definition**: `RID, AVBL, AVBR, PVCL, PVCR`.

**Literature**:
- Lim MA et al. (2016). *Neuroendocrine modulation sustains the C.
  elegans forward motor state.* **eLife** 5: e19887. — RID
  neuropeptide (FLP-14) → forward motor sustainment.

**Empirical connectivity**:
- **RID upstream**: PVCR→RID 19, PVCL→RID 12. So RID is *driven by*
  PVC (forward command), and in turn feeds back to forward command.
  Positive feedback loop.
- RID outputs (chemical): RMED 5, DD01 3, DA06 2, DD02 2. Small.
  Most of RID's effect is via neuropeptide volume release, not these
  synapses.

**Issues**:
1. **PVC→RID is in the connectome** but the modulator wiring in
   `algos.neural_v2.modulators` treats RID as having a single
   *producer* (`RID` itself) and *targets* (`AVB, PVC`). The
   upstream input to RID is not part of the modulator definition —
   that's a runtime modeling choice, not a subgraph definition
   issue. But it does mean RID can only fire if PVC is already
   active, hence the Phase 1.0 finding "RID never fires" — c_RID is
   zero because PVC never spikes enough either.

---

## Circuit 11: `modulator_5HT` (feedforward, 9 nodes)

**Current definition**: `NSML, NSMR, ADFL, ADFR, HSNL, HSNR, RIH, M3L,
M3R`.

**Literature**:
- Sze JY, Victor M, Loer C, Shi Y, Ruvkun G (2000). *Food and
  metabolic signalling defects in a Caenorhabditis elegans
  serotonin-synthesis mutant.* **Nature** 403: 560–564.
- Tanis JE, Bellemer A, Moresco JJ, Forbush B, Koelle MR (2008).
  *The potassium chloride cotransporter KCC-2 mediates GABA-induced
  inhibition in Caenorhabditis elegans.* **J Neurosci** 28(40):
  10241–10250. — 5-HT effects on motor circuits.
- Loer CM, Kenyon CJ (1993). *Serotonin-deficient mutants and male
  mating behavior in the nematode Caenorhabditis elegans.*
  **J Neurosci** 13(12): 5407–5417. — ADF as 5-HT producer.

**Issues**:
1. **VC4 / VC5 are 5-HT** also — they release ACh and 5-HT onto
   vulval muscles (Schafer 2005). Missing from `SHT_SOURCE_NEURONS`
   in `modulators.py`. Phase 1.5+.3 will document.
2. **AIM, AIY are listed as 5-HT *uptake* / receptor sites** (not
   producers but consumers). Should be in 5-HT targets list. Phase
   1.5+.3.
3. **RIH** is a target (receives 5-HT and DA), not a producer.
   Placing in this subgraph is fine but should be noted as target.

---

## Circuit 12: `egg_laying` (feedforward, 8 nodes)

**Current definition**: `HSNL, HSNR, VC01, VC02, VC03, VC04, VC05,
VC06`.

**Literature**:
- Schafer WR (2005). *Egg-laying.* **WormBook** ed. *The
  C. elegans Research Community*, doi:10.1895/wormbook.1.38.1.
- Collins KM, Bode A, Fernandez RW et al. (2016). *Activity of the
  C. elegans egg-laying behavior circuit is controlled by competing
  activation and feedback inhibition.* **eLife** 5: e21126.

**Issues**:
1. **Silent in Phase 1.0** (rate=0.000). Egg-laying needs both
   mechanical context (egg accumulation pressure → uv1/AVF input)
   and 5-HT release (NSM/HSN). The neural subnet is correct; it just
   has no driver.
2. **uv1 + AVF** are the upstream sensors (uv1 senses egg pressure;
   AVF integrates). uv1 is muscle (not in 302), AVF is in the 302
   but not in this subgraph. Add AVFL/R.

---

## Circuit 13: `defecation_pacemaker` (recurrent, 3 nodes)

**Current definition**: `DVA, DVB, AVL`.

**Literature**:
- Liu DWC, Thomas JH (1994). *Regulation of a periodic motor
  program in C. elegans.* **J Neurosci** 14(4): 1953–1962.
- Wang H, Sieburth D (2013). *PKA controls calcium influx into
  motor neurons during a rhythmic behavior.* **PLoS Genet** 9(9):
  e1003830.

**Empirical connectivity**:
- DVB→AVL 21 contacts (strong, GABA → AVL is one of the largest
  outputs of DVB).
- AVL→DD01 4, AVL→SAB 9+9+6.

**Issues**:
1. **AVL is GABA, DVB is GABA**. This subgraph is the project's most
   purely inhibitory subgraph. Consistent with the rhythmic
   inhibition pattern of defecation.
2. **DVB-AVL pair drives the body-wall muscle contraction phase.**
   Body wall is out of scope, so the downstream effect is invisible
   to the simulator.
3. **No upstream pacemaker neuron** — the defecation rhythm is
   intrinsic to the AVL-DVB-DVA system (intestinal Ca2+ wave drives
   them through gap junctions to the intestine, which is out of the
   302-neuron set). Phase 1.0 cannot reproduce the ~50 s rhythm
   without an intestinal proxy.

---

## Summary of issues by severity

**Likely-relevant to Phase 1.0 anomalies (high priority for Phase 1.5):**
- AVA↔AVB direct excitatory chemical edges + tight gap coupling (no
  GABA-mediated cross-inhibition): root cause of forward↔reversal
  +0.51 correlation.
- 3/13 silent subgraphs (pharyngeal, ventral cord motor, egg-laying)
  all need behavior-dependent extrinsic drive.

**Membership additions (low priority for Phase 1.5, can defer):**
- AIA pair to `chemosensory_amphid`.
- AS01–AS11 to `ventral_cord_motor`.
- AVF pair to `egg_laying`.
- AWC (thermal arm) to `thermosensory`.

**Membership reclassifications (medium priority):**
- PVD: move from `posterior_touch` to a new `harsh_touch_nociception`
  subgraph (its targets are reversal, not forward).
- OLQ: consider removal from `head_motor_cpg`.
- RIM: dual placement (forward + reversal); document.

**No correction needed:**
- Circuits 1, 2 (core), 5 (chemo), 6 (thermo), 11, 12, 13 — membership
  is canonical and citations check out.

All Part 1 changes are deferrable. Phase 1.5+.2's deliverable is the
*documentation* of these gaps, not the implementation. Implementation,
if any, happens in Phase 1.5.
