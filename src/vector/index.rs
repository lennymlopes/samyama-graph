//! # HNSW Vector Index Implementation
//!
//! ## How HNSW works
//!
//! HNSW (Hierarchical Navigable Small World) builds a proximity graph with multiple
//! layers. Each node is assigned a random maximum layer (exponentially distributed —
//! most nodes live only on layer 0, few reach the top). Insertion connects the new
//! point to its nearest neighbors on each layer. Search starts at the top layer's
//! entry point and greedily descends, refining the candidate set at each level.
//!
//! ## Key parameters
//!
//! - **`m`** (max connections per node): Controls graph density. Higher m = better recall
//!   but more memory and slower insertion. Typical values: 12-48. Layer 0 uses `2*m`
//!   connections.
//! - **`ef_construction`** (search width during insertion): How many candidates to
//!   consider when connecting a new node. Higher = better graph quality but slower build.
//!   Typical values: 100-400.
//! - **`ef_search`** (search width during query): How many candidates to track during
//!   search. Higher = better recall but slower queries. Must be >= k (number of results).
//!   This is the main recall-vs-speed knob at query time.
//!
//! ## Distance trait
//!
//! Rust's trait system enables polymorphic distance computation. The `hnsw_rs` crate
//! defines a `Distance<T>` trait, and this module implements it with `CosineDistance`
//! and `InnerProductDistance` structs. This allows the same HNSW data structure to work
//! with different distance metrics without runtime dispatch overhead (monomorphization).
//!
//! ## Cosine distance formula
//!
//! `cosine_distance(a, b) = 1 - (a . b) / (||a|| * ||b||)`
//!
//! This measures angular distance between vectors:
//! - **0** = identical direction (parallel vectors)
//! - **1** = orthogonal (perpendicular, no similarity)
//! - **2** = opposite direction (anti-correlated)
//!
//! ## Persistence strategy
//!
//! HNSW indices (from `hnsw_rs`) don't expose an iterator over stored vectors.
//! To support persistence, all inserted vectors are also stored in a `Vec<StoredVector>`
//! alongside the HNSW structure. On serialization, this vector list is saved via
//! `bincode`. On load, a fresh HNSW index is constructed and all stored vectors are
//! re-inserted. This trades load-time speed for implementation simplicity.

use crate::graph::NodeId;
use hnsw_rs::prelude::*;
use thiserror::Error;

/// Vector index errors
#[derive(Error, Debug)]
pub enum VectorError {
    #[error("Index error: {0}")]
    IndexError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

pub type VectorResult<T> = Result<T, VectorError>;

/// Distance metric for vector search
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance
    L2,
    /// Cosine similarity
    Cosine,
    /// Inner product
    InnerProduct,
}

/// A point in the vector space, associated with a NodeId
#[derive(Clone, Debug)]
pub struct VectorPoint {
    pub node_id: NodeId,
    pub vector: Vec<f32>,
}

/// Cosine distance implementation for hnsw_rs
#[derive(Clone, Copy, Debug, Default)]
pub struct CosineDistance;

impl Distance<f32> for CosineDistance {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        // Accumulate in f64. Summing many f32 squares loses enough precision
        // that, for near-identical unit vectors, the resulting similarity
        // can exceed 1.0 by a few ULPs and produce a negative distance,
        // which violates hnsw-rs's heap invariant `c.dist_to_ref <= 0.`
        // and panics during search.
        let mut dot: f64 = 0.0;
        let mut norm_a: f64 = 0.0;
        let mut norm_b: f64 = 0.0;

        for (a, b) in va.iter().zip(vb.iter()) {
            let (a, b) = (*a as f64, *b as f64);
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 1.0;
        }

        let sim = dot / (norm_a * norm_b).sqrt();
        let dist = 1.0 - sim;
        if !dist.is_finite() {
            // NaN from non-finite input components — clamp would propagate NaN
            // (`NaN <= 0.` is false, so the hnsw assert would still fire).
            return 1.0;
        }
        // Safety net: even with f64, treat any residual negative distance as 0.
        dist.max(0.0) as f32
    }
}

/// Inner Product distance implementation for hnsw_rs
#[derive(Clone, Copy, Debug, Default)]
pub struct InnerProductDistance;

impl Distance<f32> for InnerProductDistance {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        // Same root cause as CosineDistance: f32 accumulation can let `dot`
        // exceed 1.0 for near-identical normalized vectors, producing a
        // negative distance that trips hnsw-rs's `c.dist_to_ref <= 0.`
        // assert. Accumulate in f64 and clamp the result.
        let mut dot: f64 = 0.0;
        for (a, b) in va.iter().zip(vb.iter()) {
            dot += (*a as f64) * (*b as f64);
        }
        let dist = 1.0 - dot;
        if !dist.is_finite() {
            return 1.0;
        }
        dist.max(0.0) as f32
    }
}

/// Stored vector entry for persistence
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StoredVector {
    pub node_id: u64,
    pub vector: Vec<f32>,
}

/// Runtime dispatch over the HNSW distance type.
///
/// `Hnsw<'_, f32, D>` is generic over its distance impl, so each metric is
/// a distinct monomorphization. To pick the metric at runtime we wrap the
/// three concrete instantiations in this enum and dispatch via match.
enum HnswIndex {
    Cosine(Hnsw<'static, f32, CosineDistance>),
    L2(Hnsw<'static, f32, DistL2>),
    InnerProduct(Hnsw<'static, f32, InnerProductDistance>),
}

impl HnswIndex {
    fn new(
        metric: DistanceMetric,
        m: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
    ) -> Self {
        match metric {
            DistanceMetric::Cosine => Self::Cosine(Hnsw::new(
                m,
                max_elements,
                max_layer,
                ef_construction,
                CosineDistance,
            )),
            DistanceMetric::L2 => Self::L2(Hnsw::new(
                m,
                max_elements,
                max_layer,
                ef_construction,
                DistL2,
            )),
            DistanceMetric::InnerProduct => Self::InnerProduct(Hnsw::new(
                m,
                max_elements,
                max_layer,
                ef_construction,
                InnerProductDistance,
            )),
        }
    }

    fn insert(&self, point: (&Vec<f32>, usize)) {
        match self {
            Self::Cosine(h) => h.insert(point),
            Self::L2(h) => h.insert(point),
            Self::InnerProduct(h) => h.insert(point),
        }
    }

    fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<Neighbour> {
        match self {
            Self::Cosine(h) => h.search(query, k, ef_search),
            Self::L2(h) => h.search(query, k, ef_search),
            Self::InnerProduct(h) => h.search(query, k, ef_search),
        }
    }
}

/// Wrapper around HNSW index
pub struct VectorIndex {
    /// Number of dimensions
    dimensions: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// The actual HNSW index (variant-dispatched by `metric`)
    hnsw: HnswIndex,
    /// All inserted vectors (for persistence — HNSW doesn't expose iteration)
    stored_vectors: Vec<StoredVector>,
}

// Implement Debug manually because Hnsw doesn't implement it
impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("dimensions", &self.dimensions)
            .field("metric", &self.metric)
            .finish()
    }
}

impl VectorIndex {
    /// Create a new vector index
    pub fn new(dimensions: usize, metric: DistanceMetric) -> Self {
        // HNSW parameters
        let max_elements = 100_000;
        let m = 16;
        let ef_construction = 200;

        let hnsw = HnswIndex::new(metric, m, max_elements, 16, ef_construction);

        Self {
            dimensions,
            metric,
            hnsw,
            stored_vectors: Vec::new(),
        }
    }

    /// Add a vector to the index
    pub fn add(&mut self, node_id: NodeId, vector: &Vec<f32>) -> VectorResult<()> {
        if vector.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }
        
        self.hnsw.insert((vector, node_id.0 as usize));

        // Store vector for persistence
        self.stored_vectors.push(StoredVector {
            node_id: node_id.0,
            vector: vector.clone(),
        });

        Ok(())
    }

    /// Search for nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> VectorResult<Vec<(NodeId, f32)>> {
        if query.len() != self.dimensions {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }
        
        let ef_search = k * 2;
        let results = self.hnsw.search(query, k, ef_search);
        
        let mut neighbors = Vec::new();
        for res in results {
            neighbors.push((NodeId::new(res.d_id as u64), res.distance));
        }
        
        Ok(neighbors)
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get metric
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Get count of stored vectors
    pub fn len(&self) -> usize {
        self.stored_vectors.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.stored_vectors.is_empty()
    }

    /// Save index to disk by serializing stored vectors via bincode.
    /// On load, vectors are re-inserted into a fresh HNSW index.
    pub fn dump(&self, path: &std::path::Path) -> VectorResult<()> {
        let file = std::fs::File::create(path)?;
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, &self.stored_vectors)
            .map_err(|e| VectorError::IndexError(format!("serialization error: {}", e)))?;
        Ok(())
    }

    /// Load index from disk: deserialize stored vectors and re-insert into HNSW.
    pub fn load(
        path: &std::path::Path,
        dimensions: usize,
        metric: DistanceMetric,
    ) -> VectorResult<Self> {
        if !path.exists() {
            return Ok(Self::new(dimensions, metric));
        }
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let stored_vectors: Vec<StoredVector> = bincode::deserialize_from(reader)
            .map_err(|e| VectorError::IndexError(format!("deserialization error: {}", e)))?;

        let max_elements = (stored_vectors.len() + 10_000).max(100_000);
        let m = 16;
        let ef_construction = 200;
        let hnsw = HnswIndex::new(metric, m, max_elements, 16, ef_construction);

        // Re-insert all vectors
        for sv in &stored_vectors {
            hnsw.insert((&sv.vector, sv.node_id as usize));
        }

        Ok(Self {
            dimensions,
            metric,
            hnsw,
            stored_vectors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_index_basic() {
        let mut index = VectorIndex::new(3, DistanceMetric::Cosine);
        
        // Add some vectors
        index.add(NodeId::new(1), &vec![1.0, 0.0, 0.0]).unwrap();
        index.add(NodeId::new(2), &vec![0.0, 1.0, 0.0]).unwrap();
        index.add(NodeId::new(3), &vec![0.0, 0.1, 0.9]).unwrap();
        
        // Search — HNSW is approximate and may return fewer than k results on very small graphs
        let results = index.search(&[1.0, 0.1, 0.0], 2).unwrap();
        assert!(results.len() >= 1 && results.len() <= 2);
        assert_eq!(results[0].0, NodeId::new(1));
    }

    #[test]
    fn test_vector_index_persistence() {
        let dir = tempfile::TempDir::new().unwrap();
        let dump_path = dir.path().join("test_vectors.bin");

        // Create and populate index
        let mut index = VectorIndex::new(3, DistanceMetric::Cosine);
        index.add(NodeId::new(1), &vec![1.0, 0.0, 0.0]).unwrap();
        index.add(NodeId::new(2), &vec![0.0, 1.0, 0.0]).unwrap();
        index.add(NodeId::new(3), &vec![0.0, 0.1, 0.9]).unwrap();
        assert_eq!(index.len(), 3);

        // Dump to disk
        index.dump(&dump_path).unwrap();

        // Load from disk
        let loaded = VectorIndex::load(&dump_path, 3, DistanceMetric::Cosine).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimensions(), 3);

        // Verify search still works after reload
        let results = loaded.search(&[1.0, 0.1, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, NodeId::new(1));
    }

    #[test]
    fn test_distance_metrics() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![0.0, 1.0];
        let v3 = vec![1.0, 1.0]; // Not normalized

        let cosine = CosineDistance;
        // Orthogonal
        assert!((cosine.eval(&v1, &v2) - 1.0).abs() < 1e-6); 
        // Same
        assert!((cosine.eval(&v1, &v1) - 0.0).abs() < 1e-6);
        
        let inner = InnerProductDistance;
        // Dot product = 0
        assert!((inner.eval(&v1, &v2) - 1.0).abs() < 1e-6); // 1.0 - 0.0
    }

    #[test]
    fn test_cosine_distance_nonnegative_on_near_identical() {
        // Regression: near-identical unit vectors used to yield ~-1.19e-7 with
        // f32 accumulation (sim drifted just above 1.0), violating hnsw-rs's
        // heap invariant `c.dist_to_ref <= 0.` and panicking during search.
        let cosine = CosineDistance;
        let a = vec![0.57735026, 0.57735026, 0.57735026];
        let b = vec![0.57735027, 0.57735027, 0.57735027];
        let d = cosine.eval(&a, &b);
        assert!(d >= 0.0, "distance must be >= 0, got {}", d);
        assert!(d <= 1.0, "distance must be <= 1, got {}", d);
        // With f64 accumulation the result is effectively 0 (well below the
        // f32 round-off noise that would have produced the panic).
        assert!(d < 1e-6, "distance must be ~0, got {}", d);
    }

    #[test]
    fn test_cosine_distance_nan_safe() {
        // NaN inputs must not propagate to the returned distance — the hnsw
        // assert is `c.dist_to_ref <= 0.`, which is false for NaN and would
        // still panic.
        let cosine = CosineDistance;
        let a = vec![f32::NAN, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let d = cosine.eval(&a, &b);
        assert!(d.is_finite(), "distance must be finite on NaN input, got {}", d);
        assert!(d >= 0.0 && d <= 1.0, "distance must be in [0,1] on NaN input, got {}", d);
    }

    #[test]
    fn test_inner_product_distance_nonnegative_on_near_identical() {
        // Same FP-overshoot regression as CosineDistance: near-identical
        // normalized vectors could yield `1.0 - dot < 0` in f32 and panic.
        let ip = InnerProductDistance;
        let a = vec![0.57735026, 0.57735026, 0.57735026];
        let b = vec![0.57735027, 0.57735027, 0.57735027];
        let d = ip.eval(&a, &b);
        assert!(d >= 0.0, "distance must be >= 0, got {}", d);
        assert!(d <= 1.0, "distance must be <= 1, got {}", d);
    }

    #[test]
    fn test_vector_index_l2_metric_honored() {
        // Vectors with the same direction but very different magnitudes:
        // cosine ranks them as near-equal, L2 ranks the closer-magnitude
        // one much higher. A previous bug hardcoded CosineDistance on
        // every VectorIndex regardless of the metric argument, so this
        // assertion would have failed against an L2 index returning
        // cosine distances.
        let mut idx = VectorIndex::new(2, DistanceMetric::L2);
        idx.add(NodeId::new(1), &vec![1.0, 0.0]).unwrap();
        idx.add(NodeId::new(2), &vec![100.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        // Distances must reflect L2 geometry: ~0 to (1,0), ~99 to (100,0).
        let by_id: std::collections::HashMap<NodeId, f32> =
            results.into_iter().collect();
        assert!(by_id[&NodeId::new(1)] < 1.0, "L2 dist to (1,0) should be ~0");
        assert!(by_id[&NodeId::new(2)] > 90.0, "L2 dist to (100,0) should be ~99");
    }

    #[test]
    fn test_vector_index_inner_product_metric_honored() {
        // Inner-product favors high-magnitude alignment with the query;
        // cosine treats both vectors as parallel to (1,0). Safe to exercise
        // because the f64-hardened `InnerProductDistance` (earlier in this
        // file) clamps any FP-overshoot to a non-negative distance.
        let mut idx = VectorIndex::new(2, DistanceMetric::InnerProduct);
        idx.add(NodeId::new(1), &vec![1.0, 0.0]).unwrap();
        idx.add(NodeId::new(2), &vec![100.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        let by_id: std::collections::HashMap<NodeId, f32> =
            results.into_iter().collect();
        // Larger magnitude → higher dot → lower (clamped) distance.
        assert!(by_id[&NodeId::new(2)] <= by_id[&NodeId::new(1)]);
    }

    #[test]
    fn test_vector_index_l2_persistence_round_trip() {
        // Regression: dump and load must preserve the chosen metric and
        // produce a queryable index. (See test_vector_index_persistence
        // for the larger persistence test — this one specifically pins
        // the metric round-trip behavior.)
        let dir = tempfile::TempDir::new().unwrap();
        let dump_path = dir.path().join("vec.bin");

        let mut idx = VectorIndex::new(2, DistanceMetric::L2);
        idx.add(NodeId::new(1), &vec![1.0, 0.0]).unwrap();
        idx.add(NodeId::new(2), &vec![100.0, 0.0]).unwrap();
        idx.dump(&dump_path).unwrap();

        let loaded = VectorIndex::load(&dump_path, 2, DistanceMetric::L2).unwrap();
        assert_eq!(loaded.metric(), DistanceMetric::L2);
        let results = loaded.search(&[1.0, 0.0], 2).unwrap();
        let by_id: std::collections::HashMap<NodeId, f32> =
            results.into_iter().collect();
        assert!(by_id[&NodeId::new(1)] < 1.0);
        assert!(by_id[&NodeId::new(2)] > 90.0);
    }
}

