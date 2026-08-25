//! Bridge finding via iterative Tarjan DFS.
//! A bridge is an edge whose removal disconnects the graph.
//! Ported from FireCore MolecularGraph.h findBridges — without class ownership.

use pgraph::{CsrAdj, Index};

/// Find all bridge edges in an undirected graph. Returns edge indices that are bridges.
/// Uses iterative Tarjan's algorithm with discovery times and low-link values.
pub fn find_bridges(csr: &CsrAdj) -> Vec<Index> {
    let nverts = csr.nverts();
    let mut disc = vec![0u32; nverts];       // discovery time, 0 = unvisited
    let mut low = vec![0u32; nverts];        // low-link value
    let mut visited = vec![false; nverts];
    let mut bridges = Vec::new();
    let mut timer = 1u32;

    // Iterative DFS stack: (vertex, parent_edge_id, neighbor_scan_pos)
    // parent_edge_id = edge index used to reach this vertex (to skip the back-edge to parent)
    let mut stack: Vec<(usize, i32, usize)> = Vec::with_capacity(nverts);

    for start in 0..nverts {
        if visited[start] { continue; }
        // Push root
        visited[start] = true;
        disc[start] = timer;
        low[start] = timer;
        timer += 1;
        stack.push((start, -1, 0));

        while let Some(&(v, parent_edge, mut pos)) = stack.last() {
            let (ns, es) = csr.neighbors(v);
            // Find next unvisited neighbor or process back-edges
            let mut found_child = false;
            while pos < ns.len() {
                let n = ns[pos] as usize;
                let e = es[pos] as i32;
                pos += 1;
                if e == parent_edge { continue; } // skip the edge we came from
                if !visited[n] {
                    // Tree edge: descend
                    visited[n] = true;
                    disc[n] = timer;
                    low[n] = timer;
                    timer += 1;
                    // Update parent's scan position before pushing child
                    let top = stack.last_mut().unwrap();
                    top.2 = pos;
                    stack.push((n, e, 0));
                    found_child = true;
                    break;
                } else {
                    // Back edge: update low[v] = min(low[v], disc[n])
                    let top = stack.last_mut().unwrap();
                    let v_idx = top.0;
                    low[v_idx] = low[v_idx].min(disc[n]);
                    // Continue scanning (don't break)
                }
            }
            if !found_child {
                // Done with v's neighbors — pop and update parent's low
                stack.pop();
                if let Some(&(p, _, _)) = stack.last() {
                    low[p] = low[p].min(low[v]);
                    // Bridge condition: low[v] > disc[p] means v cannot reach above p
                    if low[v] > disc[p] {
                        // Find the edge id connecting p to v
                        let (pns, pes) = csr.neighbors(p);
                        for i in 0..pns.len() {
                            if pns[i] as usize == v {
                                bridges.push(pes[i]);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    bridges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency::build_csr_adj;

    #[test]
    fn test_no_bridges_triangle() {
        // Triangle: no bridges (every edge is in a cycle)
        let edges: Vec<[Index; 2]> = vec![[0, 1], [1, 2], [2, 0]];
        let csr = build_csr_adj(3, &edges);
        let bridges = find_bridges(&csr);
        assert!(bridges.is_empty(), "triangle should have no bridges, got {:?}", bridges);
    }

    #[test]
    fn test_all_bridges_path() {
        // Path 0-1-2-3: every edge is a bridge
        let edges: Vec<[Index; 2]> = vec![[0, 1], [1, 2], [2, 3]];
        let csr = build_csr_adj(4, &edges);
        let bridges = find_bridges(&csr);
        assert_eq!(bridges.len(), 3, "path should have 3 bridges, got {:?}", bridges);
        // All 3 edges should be bridges
        let bridge_set: std::collections::HashSet<Index> = bridges.into_iter().collect();
        for e in 0..3 { assert!(bridge_set.contains(&(e as Index)), "edge {e} should be a bridge"); }
    }

    #[test]
    fn test_mixed_graph() {
        // 0-1-2-0 (triangle) + 2-3 (bridge) + 3-4-5-3 (triangle)
        // Only edge 2-3 (index 3) is a bridge
        let edges: Vec<[Index; 2]> = vec![
            [0, 1], [1, 2], [2, 0],  // triangle 0,1,2
            [2, 3],                   // bridge
            [3, 4], [4, 5], [5, 3],   // triangle 3,4,5
        ];
        let csr = build_csr_adj(6, &edges);
        let bridges = find_bridges(&csr);
        assert_eq!(bridges, vec![3], "only edge 3 should be a bridge, got {:?}", bridges);
    }

    #[test]
    fn test_isolated_vertices() {
        // 0-1 (bridge) + isolated 2, 3
        let edges: Vec<[Index; 2]> = vec![[0, 1]];
        let csr = build_csr_adj(4, &edges);
        let bridges = find_bridges(&csr);
        assert_eq!(bridges, vec![0]);
    }
}
