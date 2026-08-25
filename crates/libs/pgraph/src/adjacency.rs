//! Adjacency builders: edge list → CSR or fixed-stride (ELL) adjacency.
//! Both produce parallel `neigh` + `edge` arrays so hot kernels don't search for bond ids.

use numtypes::{CsrAdj, FixedAdj, Index};

/// Build CSR adjacency from an edge list. Undirected: each edge appears in both endpoints.
/// Pattern: count → prefix → scatter (same as FireCore MolecularGraph::makeNeighbors).
pub fn build_csr_adj(nverts: usize, edges: &[[Index; 2]]) -> CsrAdj {
    // 1. Count degree per vertex
    let mut counts = vec![0u32; nverts];
    for &[a, b] in edges {
        counts[a as usize] += 1;
        counts[b as usize] += 1;
    }
    // 2. Prefix sum → offsets (nverts + 1)
    let mut offsets = Vec::with_capacity(nverts + 1);
    offsets.push(0);
    let mut acc = 0u32;
    for &c in &counts {
        acc += c;
        offsets.push(acc);
    }
    let total = acc as usize;
    let mut neigh = vec![0 as Index; total];
    let mut edge  = vec![0 as Index; total];
    // 3. Scatter: use a running cursor per vertex (copy of offsets)
    let mut cursor = offsets[..nverts].to_vec();
    for (e, &[a, b]) in edges.iter().enumerate() {
        let ai = a as usize;
        let bi = b as usize;
        let pos_i = cursor[ai] as usize; cursor[ai] += 1;
        let pos_j = cursor[bi] as usize; cursor[bi] += 1;
        neigh[pos_i] = b; edge[pos_i] = e as Index;
        neigh[pos_j] = a; edge[pos_j] = e as Index;
    }
    CsrAdj { offsets, neigh, edge }
}

/// Error from `build_fixed_adj`: a vertex has degree exceeding the fixed stride K.
#[derive(Debug)]
pub struct DegreeOverflow {
    pub vertex: usize,
    pub degree: usize,
    pub max_k: usize,
}

impl std::fmt::Display for DegreeOverflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "build_fixed_adj<{}>: vertex {} has degree {} > K={}",
            self.max_k, self.vertex, self.degree, self.max_k)
    }
}
impl std::error::Error for DegreeOverflow {}

/// Build fixed-stride (ELL) adjacency from an edge list. Undirected.
/// Fails loud on degree > K — never truncates (per design doc §5.1).
pub fn build_fixed_adj<const K: usize>(nverts: usize, edges: &[[Index; 2]]) -> Result<FixedAdj<K>, DegreeOverflow> {
    // Count degrees first to detect overflow before partial insertion
    let mut counts = vec![0usize; nverts];
    for &[a, b] in edges {
        counts[a as usize] += 1;
        counts[b as usize] += 1;
    }
    for (v, &d) in counts.iter().enumerate() {
        if d > K {
            return Err(DegreeOverflow { vertex: v, degree: d, max_k: K });
        }
    }
    // Scatter into fixed rows
    let mut adj = FixedAdj::<K>::new(nverts);
    for (e, &[a, b]) in edges.iter().enumerate() {
        let ai = a as usize;
        let bi = b as usize;
        adj.neigh.push(ai, b as i32).expect("push neigh: overflow checked above");
        adj.edge.push(ai, e as i32).expect("push edge: overflow checked above");
        adj.neigh.push(bi, a as i32).expect("push neigh: overflow checked above");
        adj.edge.push(bi, e as i32).expect("push edge: overflow checked above");
    }
    Ok(adj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numtypes::INVALID;

    #[test]
    fn test_csr_simple() {
        // Triangle: 0-1, 1-2, 2-0
        let edges: Vec<[Index; 2]> = vec![[0, 1], [1, 2], [2, 0]];
        let csr = build_csr_adj(3, &edges);
        assert_eq!(csr.nverts(), 3);
        assert_eq!(csr.degree(0), 2);
        assert_eq!(csr.degree(1), 2);
        assert_eq!(csr.degree(2), 2);
        // Verify neighbors of 0 are {1, 2}
        let (ns, _) = csr.neighbors(0);
        let ns_set: std::collections::HashSet<Index> = ns.iter().copied().collect();
        assert_eq!(ns_set, [1, 2].into_iter().collect());
    }

    #[test]
    fn test_csr_empty() {
        let csr = build_csr_adj(5, &[]);
        assert_eq!(csr.nverts(), 5);
        for v in 0..5 { assert_eq!(csr.degree(v), 0); }
    }

    #[test]
    fn test_fixed_adj_simple() {
        let edges: Vec<[Index; 2]> = vec![[0, 1], [1, 2], [2, 0]];
        let adj = build_fixed_adj::<4>(3, &edges).unwrap();
        assert_eq!(adj.nverts(), 3);
        assert_eq!(adj.degree(0), 2);
        assert_eq!(adj.degree(1), 2);
        assert_eq!(adj.degree(2), 2);
        // Verify entries packed first, remainder INVALID
        for v in 0..3 {
            let row = adj.neigh.row(v);
            assert!(row[0] != INVALID);
            assert!(row[1] != INVALID);
            assert_eq!(row[2], INVALID);
            assert_eq!(row[3], INVALID);
        }
    }

    #[test]
    fn test_fixed_adj_overflow() {
        // Star graph: vertex 0 has 5 neighbors, K=4 → overflow
        let edges: Vec<[Index; 2]> = vec![[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]];
        let result = build_fixed_adj::<4>(6, &edges);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.vertex, 0);
        assert_eq!(err.degree, 5);
        assert_eq!(err.max_k, 4);
    }

    #[test]
    fn test_csr_matches_fixed_adj() {
        // Random-ish graph: path 0-1-2-3-4 plus chord 0-3
        let edges: Vec<[Index; 2]> = vec![[0, 1], [1, 2], [2, 3], [3, 4], [0, 3]];
        let csr = build_csr_adj(5, &edges);
        let adj = build_fixed_adj::<4>(5, &edges).unwrap();
        for v in 0..5 {
            assert_eq!(csr.degree(v), adj.degree(v), "degree mismatch at vertex {v}");
            // Collect neighbor sets from both representations
            let (csr_ns, _) = csr.neighbors(v);
            let csr_set: std::collections::HashSet<Index> = csr_ns.iter().copied().collect();
            let fixed_set: std::collections::HashSet<i32> = adj.neigh.row(v).iter()
                .copied().filter(|&x| x != INVALID).collect();
            let csr_set_i32: std::collections::HashSet<i32> = csr_set.iter().map(|&x| x as i32).collect();
            assert_eq!(csr_set_i32, fixed_set, "neighbor set mismatch at vertex {v}");
        }
    }
}
