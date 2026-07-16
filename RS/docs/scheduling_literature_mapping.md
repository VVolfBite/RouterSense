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
| `gmwd` | residual max-weight matching and residual subtraction | paper-specific photonic/compute model | GMWD-style |
| `fast_stage` | one-to-one stages and BvN-derived priority | intra-server rebalance, server/NIC hierarchy | FAST-Stage / FAST-inspired |
| `aurora_order` | fixed-placement endpoint-pressure ordering | placement and heterogeneous-cluster optimization | Aurora-inspired ordering |
| `rsbc` | barrier and release-gain scheduling | none; RouterSense original | RSBC |
| `greedy_control` | greedy maximal matching | not applicable | Greedy Control |

The mapping level is exported in every plan contract and evaluation record.
