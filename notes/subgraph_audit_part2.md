# Subgraph audit, Part 2: missing subgraphs + bridges (Phase 1.5+.2)

> Companion to `notes/subgraph_audit_part1.md`. Identifies functional
> circuits absent from the current 13-subgraph decomposition and
> evaluates how subgraphs connect to each other ("bridges"). No code
> changes — feeds Phase 1.5+.4 (`data_audit.md`) and Phase 1.5 design.

---

## Section A: Missing subgraph candidates

### A.1 `inhibitory_command_gate` — the most consequential missing circuit

**Rationale**: Phase 1.0 reports forward ↔ reversal command pools at
**r = +0.51** when biology demands anti-correlation. The connectome's
explanation is structural: AVA ↔ AVB cross-edges are both excitatory
(see `subgraph_audit_part1.md` §1), so without an active inhibitory
gate the two pools synchronize under any common drive. The biology's
solution involves at least three GABAergic / inhibitory neurons that
are currently in NO subgraph:

- **RIS** (GABAergic; main sleep/quiescence + global command-pool
  inhibitor). Top outputs: RIBR 18, AVER 18, AVEL 17, RMDR 16,
  RIMR 15, AVKR 13 — directly inhibits both head motor and command
  circuit. Steuer Costa W et al. 2019, *Neuron* 102: 1185, document
  RIS as the master sleep / quiescence neuron.
- **AVL** (GABAergic; mostly defecation but also small outputs to
  AVE, DD01, RIS). Already in `defecation_pacemaker` but its broader
  inhibitory role overlaps command pool.
- **DD/VD class** (GABAergic motor; in `ventral_cord_motor` but
  Phase 1.0 has the whole motor pool silent).
- **RMED/V/L/R** (GABAergic; in `head_motor_cpg`). These gate head
  motor amplitude; only weak coupling to command pool (RMED→SMDDR 4).

**Proposed membership** (Phase 1.5):
```
inhibitory_command_gate (recurrent or just a tagged inhibitory hub):
  RIS, AVL, DVB,
  RIH (target / amplifier),
  ALA (sleep-promoting; outputs PVDL/R 75 each, AVEL 8, AVER 4),
  RMED, RMEV, RMEL, RMER  (overlap with head_motor_cpg)
```

**Empirical support**:
- ALA is the FLP-13 quiescence neuron (Nelson et al. 2014, *eLife*
  3: e02638). Has very large outputs to PVD (75 contacts each!) and
  modest outputs to AVE. Currently in no subgraph.
- DVB → AVL chain (21 contacts) gives a clean GABA → GABA inhibitory
  cascade.

**Priority for Phase 1.5**: **HIGH** — most direct route to fixing
the forward↔reversal anomaly.

---

### A.2 `aerotaxis` — O2/CO2 chemosensation (currently missing)

**Rationale**: ASE/AWC/AWA in `chemosensory_amphid` handle salt and
volatile attractants. They do not cover gas sensing. C. elegans has a
distinct aerotaxis circuit:

- **URX, AQR, PQR** — O2 sensing (Coates & de Bono 2002 *Nature* 419:
  925; Cheung et al. 2005 *Cell* 121: 645).
- **BAG (BAGL/R)** — CO2 sensing (Bretscher et al. 2008 *PNAS* 105:
  8044).

**Empirical connectivity**:
- URXL → RIAL 15, AUAL 14, AVEL 9, RICL 8 — goes to head + reversal.
- URXR → AUAR 21, RIAR 20, RMDR 8 — similar.
- AQR → AVBR 12, AVBL 7, AVAR 6 — drives forward + reversal command.
- PQR → AVAR 26, AVAL 23, AVDR 20, AVDL 15 — strong driver of
  reversal command.
- BAGL → RIBR 19, RIAR 18, AVER 7, RID 3 — drives integrator + RID.
- BAGR → RIBL 22, RIAL 18, AVEL 8 — similar.

Notably **BAG → RID**: BAG is one of RID's upstream drivers (3 + 3
contacts). Aerotaxis sub-network influence on RID is non-trivial and
currently invisible to the simulator (BAG fires only if there's CO2
sensory drive, which we don't supply).

**Proposed membership**:
```
aerotaxis (feedforward):
  URXL, URXR, AQR, PQR, BAGL, BAGR,
  AUAL, AUAR,        (downstream of URX)
  RIA L/R,           (premotor convergence — overlap with chemo/thermo)
  RIB L/R,           (integrator — overlap with forward_command)
  AVE L/R            (overlap with reversal_command)
```

**Priority for Phase 1.5**: **MEDIUM** — relevant because it's one of
the few currently-missing pathways into the RID modulator pool, and
Phase 1.5 will need stimulus channels for aerotactic behavior.

---

### A.3 `sleep_quiescence` — ALA/RIS quiescence network

**Rationale**: C. elegans has well-defined sleep states (lethargus,
stress-induced sleep). Two neurons are necessary and sufficient:

- **RIS** (FLP-11 neuropeptide release; required for both lethargus
  and stress-induced sleep — Turek et al. 2013 *Curr Biol* 23: 2215).
- **ALA** (FLP-13 release; stress-induced sleep — Nelson et al.
  2014 *eLife* 3: e02638; Hill et al. 2014 *Curr Biol* 24: 2399).

**Proposed membership**:
```
sleep_quiescence (recurrent / feedforward hybrid):
  ALA, RIS,
  AVKL, AVKR (FLP-1 release; lethargus quiescence — Chen et al.
              2016 *Cell* 165: 1659),
  PVT (sleep-related interneuron),
  RIH (5-HT receiver; sleep modulation)
```

**Empirical connectivity**: ALA→PVDL/R 75 contacts each = the
strongest in this set. ALA also outputs to AVE pair (8+4) and ASJ.
PVT is in `algos.graph.loader::DEFAULT_MODULATOR_NEURONS` but no
subgraph yet.

**Priority for Phase 1.5**: **MEDIUM-LOW** — important biologically
but Phase 1.5 (basic body+env) won't induce sleep states. Could
wait until Phase 2.

---

### A.4 `harsh_touch_nociception` — separation of gentle vs harsh touch

**Rationale**: `posterior_touch` currently bundles PLM (gentle touch
→ forward) with PVD (harsh touch → *reversal*). The two pathways
have opposite behavioral outputs. Empirically:
- PVDL/R → top targets are PVCL 8 and AVAL/AVAR 5 (reversal command),
  not AVB.

**Proposed membership**:
```
harsh_touch_nociception (feedforward):
  PVDL, PVDR,        (mechanical nociception)
  ASHL, ASHR,        (chemical nociception)
  ADLL, ADLR,        (osmotic/odor nociception — Hilliard 2002)
  FLPL, FLPR,        (anterior harsh touch — Chalfie 1985)
  AVAL, AVAR,        (reversal command — overlap with reversal_command)
  AVDL, AVDR         (overlap with reversal_command, anterior_touch)
```

**Priority for Phase 1.5**: **LOW** — could be a Phase 1.5 cleanup
since it just rearranges existing nodes. But it'd improve the
behavioral interpretability of touch responses.

---

### A.5 `cross_modal_integration` — AIA/AIM/AIN hub

**Rationale**: Phase 1.5+ should consider whether sensory integration
happens at named hub neurons. Three first-order interneurons that
appear in NO existing subgraph:

- **AIA (AIAL/R)** — receives ASE, AWC, ASK, AFD. Major integrator
  for chemotaxis decisions (Larsch et al. 2015 *Cell* 161: 215).
- **AIM (AIML/R)** — receives ASJ, multiple modalities; SPARC
  effects (Komuniecki 2014 *J Neurochem* 130: 731).
- **AIN (AINL/R)** — multi-modal hub, thermal memory contributions
  (Beverly et al. 2011 *J Neurosci* 31: 11718).

**Proposed membership**:
```
cross_modal_integration (recurrent / hub):
  AIAL, AIAR,
  AIML, AIMR,
  AINL, AINR,
  RIGL, RIGR,    (also a cross-modal hub per Hilliard 2005)
```

Could alternatively be folded into an expanded
`chemosensory_amphid`.

**Priority for Phase 1.5**: **LOW** — these neurons are in the
graph; if their inclusion in a subgraph matters depends on the
specific Phase 1.5 experiments. Document for awareness.

---

### A.6 Summary of missing subgraphs

| candidate | priority | n_nodes | key biology |
|---|---|---:|---|
| inhibitory_command_gate | HIGH | ~10 | RIS/ALA/AVL — relieves AVA/AVB +0.51 |
| aerotaxis | MEDIUM | ~14 | URX/AQR/PQR/BAG — drives RID, missing sensory channel |
| sleep_quiescence | MEDIUM-LOW | ~5 | ALA/RIS/AVK — needed for behavioral state |
| harsh_touch_nociception | LOW | ~10 | PVD/ASH/ADL/FLP — recategorization |
| cross_modal_integration | LOW | ~8 | AIA/AIM/AIN — sensory hubs |

**Recommendation for Phase 1.5**: implement `inhibitory_command_gate`
first; defer the rest until specific experiments demand them.

---

## Section B: Bridge connections between subgraphs

Phase 1.0's subgraphs are implemented as *views* — a node belongs to
multiple subgraphs and its V is shared (verified by
`test_subgraph_overlap_nodes_share_V_across_subgraphs`). Between
subgraphs that don't share nodes, signal flows through *bridge edges*
on the full graph but the subgraph machinery doesn't represent these
bridges explicitly. This section catalogues the major bridges so
Phase 1.5 can decide whether to model them.

### B.1 Sensory → Command bridges

The chemosensory → command bridge is mostly through `RIA → AVE`,
`AIB → AVB`, and `AIB → RIM → forward_command`:

- AIBL → AVBL 21 (chemosensory_amphid → forward_command)
- AIBR → AVBR 13
- AIBL → RIMR 56, AIBR → RIML 47 (chemo → RIM which is in fwd_cmd)
- AIZ → AVE: AIZL → AVER 26, AIZR → AVEL 28 (chemo → reversal)

**Status**: handled by node overlap (AIB and RIA are in both
`chemosensory_amphid` and either `forward_command` or
`head_motor_cpg`). The functional flow is in the graph; the bridge
itself doesn't need separate representation.

### B.2 Command → Motor bridges

This is a **broken bridge** in Phase 1.0:
- `forward_command` → `ventral_cord_motor` happens via gap junctions
  AVB↔B-class, but Cook 2019 doesn't tag rectification (see
  `edge_sign_audit.md` §4) so the bridge fires symmetrically in both
  directions — which probably contributes to the silent motor pool
  (B-class V leaks back into AVB through the symmetric gap,
  preventing AVB from reaching threshold to drive forward locomotion).
- `reversal_command` → `ventral_cord_motor` via AVA↔A-class is
  similarly affected.

**Status**: connectome data exists but symmetric assumption breaks
the functional bridge. Phase 1.5 should address (see
`edge_sign_audit.md` §4.4 recommendation).

### B.3 Modulator → Target bridges

Phase 1.0's modulator system bridges sub-graphs via the
`ModulatorBank.apply_threshold_modulation` mechanism (multiplicative
threshold scaling). The currently wired bridges:
- `modulator_RID` → forward command pool (AVB, PVC) with
  sensitivity = −0.5 (excite).
- `modulator_5HT` → forward command (AVB, AIB) with sensitivity =
  +0.4 (suppress), and → pharyngeal (M3, MI, I1) with sensitivity =
  −0.4 (excite).

**Status**: bridges exist in code but produce no observable effect in
Phase 1.0 (c_RID = 0; c_5HT ≈ 0.07). Cause is upstream: producer
neurons aren't firing. The bridge mechanism is sound; the input is
missing.

### B.4 Sensory → Modulator bridges

Empirically present in the connectome but not represented in any
subgraph or modulator wiring:
- BAG → RID 3 contacts each. CO2 sensing → forward-promoting
  modulator (would activate forward when CO2 drops, biologically
  sensible).
- URX → RIC (octopamine producer): URXL → RICL 8, URXR → RICR (need
  to check).
- ADF → broader sensory: ADF is itself a 5-HT source AND a chemosensor.
  The producer-target distinction blurs here.

**Status**: these bridges are LATENT — they exist in the connectome
but neither the subgraph definitions nor the modulator wiring exposes
them. Phase 1.5 should consider explicit modeling, especially for
BAG → RID (would give RID a sensory-driven activation route, fixing
Phase 1.0's "c_RID = 0" obstruction).

### B.5 Pharynx ↔ Soma bridges

Pharyngeal CPG is anatomically isolated, but a small number of bridge
edges connect it to the soma:
- **NSM** — bridges pharynx to body (5-HT distribution).
- **MI** — small somatic projections.
- **M1 → motor neurons** — minor outputs.
- **RIP** (pharynx-soma intercalated pair) — main anatomical
  interface (mentioned in Albertson & Thomson 1976).

**Status**: NSM is in `modulator_5HT` and `pharyngeal_cpg` (overlap
node). RIP is in no subgraph. For Phase 1.0 this doesn't matter
because pharynx is silent. For Phase 1.5, if a body-state proxy
drives the pharyngeal pump, the RIP bridge becomes the channel
through which feeding affects head/locomotion (the food-induced
quiescence loop).

### B.6 Egg-laying bridges

- **HSN** is in `modulator_5HT` and `egg_laying` — overlap node OK.
- **AVF** (currently in no subgraph) is the upstream integrator that
  takes egg-pressure → HSN modulation. Should be in `egg_laying`
  per `subgraph_audit_part1.md` §12.
- **uv1 cells** — vulval mechanoreceptors — are not in the 302 set.
  No bridge possible without a body model.

---

## Section C: Bridge summary

| bridge type | currently modeled? | status |
|---|---|---|
| Sensory → Command (chemo → AVB/AVE) | yes (overlap nodes) | ✓ works |
| Command → Motor (AVB ↔ B-class) | yes (symmetric gap) | ✗ degraded by symmetric assumption |
| Modulator → Target (RID, 5-HT → ...) | yes (threshold mod) | mechanism sound, inert without driver |
| Sensory → Modulator (BAG → RID; URX → RIC) | NO | latent — likely worth adding in Phase 1.5 |
| Pharynx ↔ Soma (NSM, RIP) | partial (NSM overlap) | OK while pharynx silent |
| Egg-laying upstream (uv1, AVF → HSN) | NO | AVF missing; uv1 needs body |

---

## Section D: What this audit recommends

Phase 1.5+.2 deliverable (this doc + Part 1) is **descriptive**.
Implementation is for Phase 1.5. Concrete actions ranked by expected
impact on Phase 1.0's three known failures:

1. **Failure: forward ↔ reversal +0.51 correlation.**
   - Highest-leverage fix: introduce `inhibitory_command_gate`
     subgraph + per-edge rectification on AVA/AVB ↔ motor gap
     junctions.
2. **Failure: 3/13 subgraphs silent (pharynx, motor, egg-laying).**
   - These need extrinsic behavioral drive. Phase 1.5's body+env
     supplies it; no subgraph definition change required.
3. **Failure: modulators inert (RID = 0, 5-HT saturates low).**
   - Add sensory→modulator bridges (BAG → RID structurally documented;
     could be added as direct chemical edge OR via a "behavioral
     modulator input" channel).

Nothing here is a code bug. Every gap is a documented modeling
choice that Phase 1.5 can revisit.
