# Results Schema

Reported machine-readable summary outputs are stored under:

```text
results/summary_public/reported/
```

The tables support direct inspection and downstream plotting or analysis.

## Summary-Output Categories

| Category | Table content |
|---|---|
| Cost–emission frontier | Absolute-cap frontier points and system composition |
| Temporal sensitivity | Representative-period sensitivity and stress-window outcomes |
| Storage lifecycle | Calendar ageing, BESS, long-duration storage, equivalent-cycle, and cost-decomposition summaries |
| Flexibility | Demand-response, thermal, EV, and combined-flexibility ablation summaries |
| Uncertainty | LHS samples, signed PRCC rankings, and uncertainty ranges |
| MAC and bottleneck | Finite-difference MAC, dual-based MAC, binding counts, and one-percent relaxation outputs |
| Spatial aggregation | k70, k100, and k150 metrics and screening outputs |
| Renewable potential | PV and wind technical-potential audit tables |
| Network abstraction | Node-interface, copperplate, and spatial-transfer summaries |
| Validity diagnostics | Run-family validity counts and constraint-relaxation diagnostics |

Each CSV header defines its fields. `data/catalog.yaml` records the path, byte size, SHA-256 checksum, role, and MIT license for every reported table.
