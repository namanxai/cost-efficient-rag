# Cost Comparison: Self-Hosted ChromaDB vs. Managed Vector DB

## Assumptions

- **Vector dimensionality:** 384 (matches `all-MiniLM-L6-v2` family; this repo's sandbox
  substitute embedding is also 384-dim for a fair comparison).
- **Storage per vector:** ~384 floats × 4 bytes = 1,536 bytes, plus metadata/index
  overhead (HNSW graph, doc text, IDs) — budgeted at ~2.5 KB/vector all-in.
- **Managed vector DB baseline:** Pinecone-style always-on pod pricing (public list
  pricing as of early 2026, approximate): a standard pod handles up to ~1M vectors
  at 768 dim (roughly ~2M at 384 dim) for around **$70/pod/month**, billed whether
  or not it's queried (this is the "always-on pods" cost the assignment flags).
- **Self-hosted baseline:** a small persistent VM/container running ChromaDB
  (embedded/self-hosted mode) on cheap block storage, e.g. a $12–24/month VPS
  (2 vCPU / 4GB RAM) covering compute, plus object/disk storage at roughly
  **$0.10/GB/month** (standard cloud SSD pricing).
- Query volume is assumed **light-to-moderate** ("lightly-queried index" per the
  prompt) — a few thousand queries/day, not a high-QPS production search engine.
  At high QPS, the calculus shifts back toward managed (see Discussion).
- No cost attributed to embedding generation (assumed amortized/one-time at ingest).

## Cost at Scale

| Vectors | Est. Storage | Self-hosted (VM + disk) | Managed DB (always-on pods) |
|---|---|---|---|
| 100,000    | ~250 MB  | **~$15/mo** (small VPS, storage negligible) | **~$70/mo** (1 pod minimum, even though only 5% utilized) |
| 1,000,000  | ~2.5 GB  | **~$18/mo** (VPS + a few GB disk) | **~$70–140/mo** (1–2 pods depending on provider's per-pod ceiling at this dim) |
| 10,000,000 | ~25 GB   | **~$35–50/mo** (bigger VM for query throughput + ~$2.50 disk) | **~$700/mo** (10+ pods, since pods cap out well under 10M vectors each) |

*(Numbers are illustrative, order-of-magnitude estimates from public pricing pages,
not a quote — the point of the table is the shape of the curve, not the exact
dollar figures.)*

## The Core Trade-off

Managed vector DB cost scales with **capacity provisioned** (pods are always-on,
sized for peak, billed regardless of query volume). Self-hosted cost scales with
**actual storage + the compute you choose to run it on** — for a lightly-queried
index, you can run it on a small always-on box or even spin it up on demand.

The gap widens hard at scale specifically because pod-based managed pricing has
step-function jumps (you buy a whole extra pod once you cross its vector/dim
ceiling), while a self-hosted disk-backed store scales closer to linearly with
actual bytes stored.

## What You Give Up Self-Hosting

- No automatic multi-region replication / high availability out of the box —
  you own uptime, backups, and failover.
- Query throughput and recall at very high QPS or very large (100M+) vector
  counts needs real tuning (sharding, ANN index parameters); managed services
  handle this for you.
- No built-in enterprise features (RBAC, audit logs, SOC2 attestations) that
  managed vendors sell as part of the package — relevant if this is a
  regulated/enterprise context.

See `README.md` → Discussion for when to switch back to managed.
