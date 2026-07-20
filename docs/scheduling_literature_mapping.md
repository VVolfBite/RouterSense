# Scheduling Literature Mapping

## Primary references

1. Yiran Lei et al., **FAST: An Efficient Scheduler for All-to-All GPU Communication**, NSDI 2026.
   FAST addresses skew using intra-server rebalancing and balanced one-to-one
   scale-out transfers on heterogeneous two-tier GPU fabrics.

2. Eliezer Amponsah and Vamsi Addanki, **Birkhoff Decompositions and Photonic
   Interconnects: Wait! Don't Forget the Compute!**, HotOptics 2026.
   The paper studies Greedy Max-Weight Decomposition (GMWD), repeatedly
   extracting maximum-weight matchings from the original residual MoE matrix to
   preserve larger expert batches and reduce fragmentation.

3. Jialong Li et al., **Optimizing Mixture-of-Experts Inference Time Combining
   Model Deployment and Communication Scheduling** (Aurora), 2024.
   Aurora combines token-transmission ordering with expert/model deployment.

4. Heng Xu et al., **RailS: Load Balancing for All-to-All Communication in
   Distributed Mixture-of-Experts Training**, 2025.
   RailS uses rail-topology symmetry and per-node LPT spraying.  RouterSense does
   not currently expose a rail topology, so no current family may be labelled
   RailS-style.

## Mechanism audit

| Family | Implemented | Missing | Allowed label |
|---|---|---|---|
| `gmwd_style_reference` | phase-local residual maximum-weight matching and common-quantum subtraction | photonic reconfiguration and profiled compute/overlap model | GMWD-style |
| `fast_stage_reference` | server-level traffic collapse, one-to-one server stages, GPU-edge realization, local idle-port fill | endpoint-mutating rebalance and two-tier pipeline cost model | FAST-style |
| `aurora_order_reference` | bottleneck-sender seeding and conflict-avoiding fixed-placement ordering | expert colocation, deployment and heterogeneous assignment | Aurora-style fixed-placement ordering |
| `islip_reference` | iterative request/grant/accept with persistent round-robin pointers | switch-cell and hardware queue timing | iSLIP-style |
| `gmwd` | RouterSense formal matching family using a GMWD-style residual core | paper-specific photonic/compute model | GMWD-style internal ablation |
| `rsbc` | barrier and release-gain scheduling | none; RouterSense original | RSBC |
| `rscf` | transitive critical-frontier and endpoint-dual scheduling | none; RouterSense original | RSCF |

The mapping level is exported in every plan contract and evaluation record.
