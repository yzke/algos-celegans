# C. elegans modulator system — full data compilation (Phase 1.5+.3)

> Sub-phase 1.5+.3 deliverable. Compiles producer / target / function
> / time-constant / reference data for every modulator family known
> to operate in the C. elegans hermaphrodite nervous system. This is
> data, not code — Phase 1.0 implements only RID (FLP-14 stand-in)
> and 5-HT (`src/algos/neural_v2/modulators.py`). Phase 1.5+ will
> decide which others to wire.
>
> Time-constant estimates are project-internal heuristics (in
> simulator ticks) chosen to be consistent with `tau_m ≫ neuron tau`
> (design §7.2). Where literature supplies absolute timescales they
> are noted; otherwise the heuristic is "small molecules ≈ 300 ticks,
> peptides ≈ 800 ticks, gas/volume ≈ 2000 ticks".

---

## Conventions

For each modulator entry below:

- **Producers** = neurons that release the molecule. All neuron names
  are verified present in our 302-neuron connectome
  (`ConnectomeData`).
- **Targets** = neurons expressing receptors / known to respond.
  Marked direction `+` (excite), `−` (inhibit), `?` (unclear).
- **Function** = behavioral / circuit-level role from primary
  literature.
- **τ_m estimate** = simulator-tick suggestion. Conservative.
- **References** = paper-level citations. WormBook chapters are noted
  but primary literature preferred.
- **Phase 1.0 status** = whether this modulator is wired in
  `algos.neural_v2.modulators.py` today.

---

## 1. Serotonin (5-HT)

### Producers (4 pairs + 2 single = 9 neurons + 1 male-specific not in herm)

| neuron | confirmed in 302? | notes |
|---|---|---|
| NSML | ✓ (#18) | primary 5-HT source; food sensing |
| NSMR | ✓ (#19) | |
| ADFL | ✓ (#32) | secondary 5-HT, chemosensory pair |
| ADFR | ✓ (#33) | |
| HSNL | ✓ (#292) | egg-laying motor neuron, ACh + 5-HT |
| HSNR | ✓ (#293) | |
| VC04 | ✓ (#297) | vulval motor, ACh + 5-HT |
| VC05 | ✓ (#298) | |
| RIH | ✓ (#107) | mostly a *target*, with some volume release per
  Loer 1993 — kept here for compactness |

### Targets (selection — full list runs into dozens)

| target | direction | notes |
|---|---|---|
| AVB (AVBL/R) | − (suppress forward) | Sze 2000; mediated by SER-1/SER-7 |
| AIB (AIBL/R) | − | Komuniecki 2014 review |
| M3 (M3L/R) | + (excite pharyngeal pump) | Tanis 2008 |
| M4 | + | Niacaris & Avery 2003 |
| MI | + | mild; Niacaris & Avery 2003 |
| I1 (I1L/R) | + | pharyngeal contribution |
| AIY (AIYL/R) | + (food-state shift) | Iwanir 2016 *Curr Biol* 26: 2446 |
| AIM (AIML/R) | + (5-HT uptake confirmed) | Jafari 2011 — **uptake target, not target of synaptic 5-HT signalling per se** |
| RIA (RIAL/R) | ? | weak receptor expression |
| Vulval muscles | + | egg-laying activation; out of 302 |

**Function summary**: 5-HT is the *food-presence* signal. High 5-HT →
slow forward locomotion (basal slowing response, Sawin Ranganathan
Horvitz 2000), suppress reversal, promote feeding (M3 excitation),
promote egg-laying (vulval muscle excitation).

**τ_m estimate**: 300–500 ticks. Small molecule; clearance by
reuptake (MOD-5, the SERT homolog; Ranganathan 2001).

**References**:
- Horvitz HR, Chalfie M, Trent C, Sulston JE, Evans PD (1982).
  *Serotonin and octopamine in the nematode Caenorhabditis elegans.*
  **Science** 216: 1012–1014. — original identification.
- Sze JY, Victor M, Loer C, Shi Y, Ruvkun G (2000). **Nature** 403:
  560–564.
- Sawin ER, Ranganathan R, Horvitz HR (2000). *C. elegans locomotory
  rate is modulated by the environment through a dopaminergic
  nervous system and by experience through a serotonergic nervous
  system.* **Neuron** 26: 619–631.
- Tanis JE et al. (2008). **J Neurosci** 28(40): 10241–10250.
- Iwanir S, Brown AS, Nagy S et al. (2016). **Curr Biol** 26(18):
  2446.
- Ranganathan R, Sawin ER, Trent C, Horvitz HR (2001). *Mutations in
  the Caenorhabditis elegans serotonin reuptake transporter MOD-5
  reveal serotonin-dependent and -independent activities of
  fluoxetine.* **J Neurosci** 21: 5871.

**Phase 1.0 status**: wired (`5HT` Modulator). Producers: NSML/R,
ADFL/R, HSNL/R. Targets: AVB, AIB (suppressive); M3, MI, I1
(excitatory). **Missing producers**: VC04, VC05. **Missing targets**:
AIY (excitation), M4, vulval muscles (out of 302).

---

## 2. Dopamine (DA)

### Producers (4 pairs = 8 neurons)

| neuron | confirmed in 302? |
|---|---|
| CEPDL, CEPDR, CEPVL, CEPVR | all ✓ |
| ADEL, ADER | ✓ |
| PDEL, PDER | ✓ |

All eight are mechanosensory cilia in addition to dopamine release —
classic dual-role neurons.

### Targets (selection)

| target | direction | notes |
|---|---|---|
| Locomotion network | − (basal slowing on food) | Sawin 2000 |
| AVA (AVAL/R) | + via DOP-1 | Chase Pepper Koelle 2004 *Nat Neurosci* 7: 1096 |
| AVE | + | Chase 2004 |
| RIM (RIML/R) | varies; DOP-1 and DOP-3 antagonistic | Ezcurra 2011 *EMBO J* 30: 1110 |
| Touch sensitivity (ALM/PLM) | habituation modulation | Kindt et al. 2007 *Nat Neurosci* 10: 1300 |

**Function**: DA mediates the **basal slowing response** — worms slow
forward locomotion when sensing a bacterial lawn (mechanical
detection by CEP/ADE/PDE cilia). Also modulates touch habituation
and learning. Bidirectional through different receptors (DOP-1
excitatory, DOP-3 inhibitory).

**τ_m estimate**: 300–500 ticks. Small molecule, reuptake via DAT-1.

**References**:
- Sulston J, Dew M, Brenner S (1975). *Dopaminergic neurons in the
  nematode Caenorhabditis elegans.* **J Comp Neurol** 163(2):
  215–226.
- Sawin Ranganathan Horvitz 2000 (Neuron) — basal slowing.
- Chase DL, Pepper JS, Koelle MR (2004). *Mechanism of extrasynaptic
  dopamine signaling in Caenorhabditis elegans.* **Nat Neurosci**
  7(10): 1096–1103.
- Kindt KS et al. (2007). *Dopamine mediates context-dependent
  modulation of sensory plasticity in C. elegans.* **Neuron** 55(4):
  662–676.
- Ezcurra M, Tanizawa Y, Swoboda P, Schafer WR (2011). *Food
  sensitises C. elegans avoidance behaviours through acute
  dopamine signalling.* **EMBO J** 30(6): 1110–1122.

**Phase 1.0 status**: **NOT wired**. CEP/ADE/PDE are tagged in
`algos.graph.loader.DEFAULT_MODULATOR_NEURONS`? Verified: no, they
are not in the 14-neuron list. For Phase 1.5+ the DEFAULT_MODULATOR_NEURONS
list should be extended.

---

## 3. Octopamine (OA)

### Producers

| neuron | confirmed | notes |
|---|---|---|
| RICL | ✓ | sole confirmed octopaminergic neuron pair in hermaphrodite |
| RICR | ✓ | |

### Targets (selection)

| target | direction | notes |
|---|---|---|
| AVA, AVE | + (promote reversal) | Mills et al. 2012 *EMBO J* 31: 667 |
| Forward locomotion network | − | Suo Kimura van der Kooy 2009 |
| Pharyngeal pumping | − (suppress feeding) | Horvitz 1982 |
| Egg-laying | − | Horvitz 1982 |

**Function**: OA is the *starvation / aroused* signal — opposite of
5-HT. Released when food is absent; suppresses feeding + egg-laying;
promotes reversal. Acts through SER-3, SER-6 receptors.

**τ_m estimate**: 300–500 ticks. Small molecule.

**References**:
- Horvitz Chalfie Trent Sulston Evans 1982 (Science) — original.
- Alkema MJ, Hunter-Ensor M, Ringstad N, Horvitz HR (2005).
  *Tyramine functions independently of octopamine in the
  Caenorhabditis elegans nervous system.* **Neuron** 46(2): 247–260.
- Suo S, Kimura Y, Van Tol HHM (2006). *Starvation induces cAMP
  response element-binding protein-dependent gene expression
  through octopamine-Gq signaling in Caenorhabditis elegans.*
  **J Neurosci** 26(40): 10082–10090.
- Mills H, Wragg R, Hapiak V et al. (2012). *Monoamines and
  neuropeptides interact to inhibit aversive behaviour in
  Caenorhabditis elegans.* **EMBO J** 31(3): 667–678.

**Phase 1.0 status**: **NOT wired**. RICL/R are in
`DEFAULT_MODULATOR_NEURONS` but no separate `octopamine` Modulator
entry exists.

---

## 4. Tyramine (TA)

### Producers

| neuron | confirmed | notes |
|---|---|---|
| RIML | ✓ | the famous case: Glu + tyramine co-release |
| RIMR | ✓ | |

Some literature also lists UV1, RIC as tyraminergic (Alkema 2005);
UV1 is muscle (out of 302). RIC's tyramine arm is the precursor to
octopamine; biosynthesis pathway TDC-1 → TBH-1 in RIC.

### Targets (selection)

| target | direction | notes |
|---|---|---|
| AVB (AVBL/R) | − (suppress forward, via LGC-55) | Pirri 2009 |
| MC (MCL/R, pharyngeal) | − (suppress pumping) | Pirri 2009 |
| RMD pair (head motor) | − (inhibit head turning) | Pirri 2009 |
| Body wall muscle | − (relax) | Pirri 2009 |

**Function**: TA is the **escape signal** — released by RIM during
the omega turn / escape response, simultaneously suppressing forward
locomotion + head turning + pumping to clear the way for the worm to
flee. Acts through LGC-55 (a tyramine-gated chloride channel) for
*fast* effects and SER-2 / TYRA-3 (GPCRs) for slow.

**τ_m estimate**: For LGC-55 mediated effects, ~50 ticks (almost
spike-fast). For GPCR effects, ~300 ticks. Phase 1.0's slow modulator
framework only accommodates the slow arm; the fast LGC-55 arm would
need direct chemical edges with sign=−1 from RIM to LGC-55-bearing
targets.

**References**:
- Alkema MJ, Hunter-Ensor M, Ringstad N, Horvitz HR (2005). **Neuron**
  46: 247–260.
- Pirri JK, McPherson AD, Donnelly JL, Francis MM, Alkema MJ (2009).
  *A tyramine-gated chloride channel coordinates distinct motor
  programs of a Caenorhabditis elegans escape response.* **Neuron**
  62(4): 526–538.
- Maguire SM, Clark CM, Nunnari J, Pirri JK, Alkema MJ (2011).
  *The C. elegans touch response facilitates escape from predacious
  fungi.* **Curr Biol** 21(15): 1326–1330.

**Phase 1.0 status**: **NOT wired**. Highest-leverage missing
modulator for resolving the forward↔reversal +0.51 anomaly: RIM is
already firing in the bare simulator (it's in `forward_command`); a
tyramine arm targeting AVB would *automatically* introduce the
mutual exclusion Phase 1.0 lacks.

---

## 5. RID neuropeptides (FLP-14 + FLP-1)

### Producer

| neuron | confirmed | notes |
|---|---|---|
| RID | ✓ | unpaired single neuron; main effector is FLP-14 |

### Targets (selection)

| target | direction | notes |
|---|---|---|
| AVB, PVC | + (sustain forward) | Lim 2016 |
| RIB | + (forward-bias integrator) | Lim 2016 |
| RMD/RME (head motor) | + (mild) | Lim 2016 |
| DA, DB cholinergic motor | + (mild) | Lim 2016 |

**Function**: RID sustains the forward locomotion state for tens of
seconds at a time. Lim 2016 shows RID firing inferred from FLP-14
release; ablation causes forward-bout shortening. RID is part of the
positive feedback loop with PVC: PVC → RID (chemical, 12-19 contacts)
→ FLP-14 → PVC and AVB excitation.

**τ_m estimate**: 800–1500 ticks (neuropeptide; slower than monoamines).

**References**:
- Lim MA, Chitturi J, Laskova V et al. (2016). *Neuroendocrine
  modulation sustains the C. elegans forward motor state.*
  **eLife** 5: e19887. doi:10.7554/eLife.19887.

**Phase 1.0 status**: **WIRED** as the `RID` Modulator with
producer={RID}, target={AVBL/R, PVCL/R}, sensitivity=−0.5 (excitatory
because parameter modulation: negative sensitivity lowers
threshold). τ_m=500 ticks default. Phase 1.0 finding: c_RID = 0
across all seeds because RID gets no drive from the bare network
(only chemical input from PVC, which is also silent).

---

## 6. FLP-1 (AVK + others)

### Producers

| neuron | confirmed | notes |
|---|---|---|
| AVKL | ✓ | main FLP-1 producer (Kim & Li 2004) |
| AVKR | ✓ | |
| PVT | ✓ | secondary FLP-1 (Chen 2016) |
| DVA | ✓ | tertiary |

### Targets

| target | direction | notes |
|---|---|---|
| Body wall muscle tone | − (relax) | Chen 2016 |
| Locomotion network (AVA) | − (suppress reversal) | Chen 2016 |

**Function**: FLP-1 mediates **lethargus quiescence** (sleep-like
state between larval molts) and contributes to body posture
maintenance. AVK is the principal source.

**τ_m estimate**: 1000–2000 ticks (peptide; sleep-state timescale).

**References**:
- Kim K, Li C (2004). *Expression and regulation of an FMRFamide-
  related neuropeptide gene family in C. elegans.* **J Comp Neurol**
  475(4): 540–550.
- Chen D, Taylor KP, Hall Q, Kaplan JM (2016). *The neuropeptides
  FLP-2 and PDF-1 act in concert to arouse Caenorhabditis elegans
  locomotion.* **Genetics** 204(3): 1151–1159. — describes a related
  PDF pathway; FLP-1 details cross-referenced.
- Kawano T et al. 2011 (Neuron) — locomotion coupling.

**Phase 1.0 status**: **NOT wired** as a modulator. AVKL/R and PVT
are tagged in `DEFAULT_MODULATOR_NEURONS` so the producer side is
flagged, but no `ModulatorBank` entry exists.

---

## 7. FLP-13 (ALA stress-induced sleep)

### Producer

| neuron | confirmed | notes |
|---|---|---|
| ALA | ✓ | sole FLP-13 producer of note |

### Targets

| target | direction | notes |
|---|---|---|
| PVDL, PVDR | + (sensory amplification?) — but ALA→PVD has 75 contacts each, the strongest in the connectome |
| AVE | − (sleep induction) |
| ASJ | − (sleep) |
| RMD pair | − |

**Function**: ALA-released FLP-13 induces stress-response quiescence
(stress-induced sleep, SIS). The connectome shows ALA→PVD
(non-modulator chemical edges) are the largest in this neuron's
output set — biology is unclear about why a sleep-promoting neuron
has 75 chemical contacts on a touch sensor.

**τ_m estimate**: 1500–3000 ticks (slow peptide; sleep timescale).

**References**:
- Nelson MD, Janssen T, Schoofs L, Raizen DM (2014). *FRPR-4 is a
  G-protein coupled neuropeptide receptor that regulates behavioral
  quiescence and posture in Caenorhabditis elegans.* **eLife** 3:
  e02638.
- Hill AJ, Mansfield R, Lopez JM, Raizen DM, Van Buskirk C (2014).
  *Cellular stress induces a protective sleep-like state in C.
  elegans.* **Curr Biol** 24(20): 2399–2405.
- Steuer Costa W, Van der Auwera P, Glock C et al. (2019).
  *A GABAergic and peptidergic sleep neuron as a locomotion
  stopwatch in Caenorhabditis elegans.* **Neuron** 102(2):
  1185–1198. — RIS counterpart.

**Phase 1.0 status**: **NOT wired**. ALA is not in
`DEFAULT_MODULATOR_NEURONS` either — even structural promotion
missing.

---

## 8. FLP-11 (RIS lethargus + sleep)

### Producer

| neuron | confirmed | notes |
|---|---|---|
| RIS | ✓ | GABAergic + FLP-11 release |

### Targets

| target | direction | notes |
|---|---|---|
| AVE, AVA | − (deep inhibition of command pool) | Turek 2013 |
| RMD pair | − (head-motor inhibition) | Turek 2013 |
| RIB | − | Turek 2013 |

**Function**: RIS is the master sleep neuron — required for both
lethargus and stress-induced sleep. Releases BOTH GABA (fast, via
UNC-49) AND FLP-11 (slow). The GABA arm is already in our W_chem (RIS
is in our 26-GABA list, sign=−1). The FLP-11 arm is not modeled.

**τ_m estimate**: 1500–3000 ticks (sleep timescale).

**References**:
- Turek M, Lewandrowski I, Bringmann H (2013). *An AP2 transcription
  factor is required for a sleep-active neuron to induce sleep-like
  quiescence in C. elegans.* **Curr Biol** 23(22): 2215–2223.
- Steuer Costa W et al. 2019 (Neuron) — as above.

**Phase 1.0 status**: chemical GABA arm WIRED via the standard
`W_chem` machinery (RIS is in the 26-GABA list). The FLP-11
modulator arm NOT wired.

---

## 9. NLP-12 (DVA proprioception)

### Producer

| neuron | confirmed | notes |
|---|---|---|
| DVA | ✓ | unique DVA-derived NLP-12; couples stretch to locomotion |

### Targets

| target | direction | notes |
|---|---|---|
| Cholinergic motor (DA, VA, DB, VB) | + (sustain wave) | Hu 2011, *Neuron* 72: 92 |
| AVA | + (modest) | Hu 2011 |

**Function**: NLP-12 is the *proprioceptive feedback peptide* — DVA
senses body-wall stretch (mechanosensation), releases NLP-12, which
sustains motor neuron firing. Critical for the proper turn-and-reset
of the locomotion wave.

**τ_m estimate**: 500–1000 ticks (intermediate; couples to motor
wave period).

**References**:
- Hu Z, Pym ECG, Babu K, Vashlishan Murray AB, Kaplan JM (2011).
  *A neuropeptide-mediated stretch response links muscle contraction
  to changes in neurotransmitter release.* **Neuron** 71(1): 92–102.

**Phase 1.0 status**: **NOT wired**. DVA is in
`DEFAULT_MODULATOR_NEURONS`. Phase 1.5+ should consider: NLP-12
needs proprioceptive (body) input, so this is properly Phase 1.5+
territory.

---

## 10. INS family (insulin-like)

### Producers

| neuron | confirmed | notes |
|---|---|---|
| AIA (AIAL/R) | ✓ | INS-1 release; chemotaxis learning |
| ASI (ASIL/R) | ✓ | DAF-7/DAF-16 dauer signaling — slow timescale |
| ASJ (ASJL/R) | ✓ | INS-related; dauer entry / exit |

### Targets

Largely diffuse / GPCR-based (DAF-2 insulin receptor is broadly
expressed). The "modulator" here operates on a near-developmental
timescale.

**Function**: INS family handles learning, dauer entry/exit, lifespan
regulation. Effects measured in hours to days. **Out of scope** for
Phase 1.0/1.5 dynamics (which operate at ms-to-second tick scale).

**Phase 1.0 status**: NOT wired and probably shouldn't be (timescale
mismatch).

---

## 11. NLP-3 (NSM additional peptide)

NSM releases 5-HT + NLP-3 + FLP-21. NLP-3 effects on feeding are
similar to 5-HT (Frooninckx 2012 review). Treating NSM as 5-HT-only
in Phase 1.0 is a reasonable simplification.

---

## 12. PDF (pigment-dispersing factor)

### Producers

| neuron | confirmed |
|---|---|
| In hermaphrodite: pdf-1 expression weaker; mainly PDF-2 | — |

PDF is the *circadian / locomotion arousal* peptide. Roles in
hermaphrodite less clear than in male (where PDF dominates mating
arousal).

**Phase 1.0 status**: NOT wired. Lower priority.

---

## Cross-modulator summary table

| modulator | producers | n_targets (canonical) | τ_m est. | Phase 1.0 status |
|---|---|---:|---:|---|
| 5-HT | NSM, ADF, HSN, VC4/5 | ~12 | 300–500 | **wired (partial)** |
| Dopamine | CEP, ADE, PDE | ~6 | 300–500 | NOT wired |
| Octopamine | RIC | ~4 | 300–500 | NOT wired |
| Tyramine | RIM | ~4 | 50 (fast) / 300 (slow) | NOT wired |
| RID/FLP-14 | RID | ~5 | 800–1500 | **wired** |
| FLP-1 | AVK, PVT, DVA | ~3 | 1000–2000 | NOT wired |
| FLP-13 | ALA | ~4 | 1500–3000 | NOT wired |
| FLP-11 | RIS | ~4 | 1500–3000 | NOT wired (GABA arm via W_chem only) |
| NLP-12 | DVA | ~6 | 500–1000 | NOT wired (needs body) |
| INS family | AIA, ASI, ASJ | many | hour+ | OUT OF SCOPE |

**Total: 10 modulator families, 5 currently wirable in dynamics (the
rest need body or are too slow); 2 wired in Phase 1.0 (5-HT, RID).**

---

## Action items for Phase 1.5+

### Immediate (Phase 1.5 design must address):

1. **Add tyramine modulator** with RIM as producer and AVB/MC/RMD as
   targets. Highest leverage for fixing forward↔reversal +0.51
   anomaly because RIM is already firing in the bare network. Two
   implementation options:
   - Fast LGC-55 arm: direct chemical edges, sign=−1, from RIM to
     AVB/MC/RMD. Bypasses the Modulator machinery.
   - Slow GPCR arm: ModulatorBank entry with producer={RIM}, targets
     above, sensitivity=+0.5 (positive → raises threshold →
     suppresses).
2. **Add dopamine modulator** with CEP/ADE/PDE producers and AVA/AVE
   targets. Promotes basal slowing; relevant when Phase 1.5 introduces
   bacterial-lawn mechanosensation.
3. **Add BAG → RID and URX → RIC sensory→modulator bridges** (see
   `notes/subgraph_audit_part2.md` §B.4).

### Defer (Phase 2+):

4. FLP-1, FLP-13, FLP-11 (need sleep-state behavioral context).
5. NLP-12 (needs proprioceptive body input).
6. INS family (timescale mismatch with Phase 1.5 simulation length).
7. PDF (hermaphrodite role unclear).

---

## Open questions (forwarded to QUESTIONS.md)

- Tyramine implementation: fast LGC-55 (chemical) vs slow GPCR
  (modulator)? Both arms exist in biology. **Q-1.5+.5**.
- Dopamine: DOP-1 vs DOP-3 antagonism — Phase 1.5 modulator
  framework only supports single-direction sensitivity. **Q-1.5+.6**.
- Co-release accounting: NSM releases 5-HT + NLP-3 + FLP-21
  simultaneously. Should they be three modulator pools or merged?
  **Q-1.5+.7**.
