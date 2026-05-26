# Upgrade note: `VectorIndex` distance metric

## What changed

Prior to this fix, `VectorIndex` (`src/vector/index.rs`) accepted a
`DistanceMetric` enum (`Cosine` / `L2` / `InnerProduct`) but always built its
underlying HNSW with `CosineDistance`. The `metric` field was stored and
serialized but never reached the HNSW. Effectively only one metric was
supported regardless of caller intent.

After this fix, the chosen metric is honored end-to-end. `Hnsw` is now
dispatched through an internal `HnswIndex` enum that holds the correct
monomorphization.

## Who is affected

Anyone with persisted vector indices whose `metadata.json` records
`"metric": "L2"` or `"metric": "InnerProduct"`. The on-disk vector list
itself is metric-agnostic (`Vec<StoredVector>` written via `bincode`), but
the *queryable HNSW* rebuilt on load was previously always cosine and is
now whatever metric was recorded.

Indices recorded as `"Cosine"` are unaffected.

## What you will observe

After upgrading and reloading an affected index:

- **L2**: nearest-neighbor results change. They are now Euclidean-correct.
  Old results were cosine distances under an L2 label.
- **InnerProduct**: same — results now reflect dot product. Requires the
  companion `f64`-hardened `CosineDistance` / `InnerProductDistance`
  (shipped together with this fix in the same PR series) so that FP
  overshoot does not trip hnsw-rs's `f.dist_to_ref >= 0.` assertion.

In both cases the *new* results are the intended semantics; the *old*
results were a silent bug.

## Action required

None for cosine indices. For L2 / InnerProduct indices: re-run any
downstream code that snapshotted nearest-neighbor results against a
loaded index. Application-level expectations tied to the old cosine
behavior under an L2 / IP label must be updated.

The on-disk file format does not change. No re-dump is necessary; the
reload path produces the correct index automatically.
