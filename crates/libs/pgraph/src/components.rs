//! Connected components via BFS. Returns a Partition assigning each vertex to a component id.

use numtypes::{CsrAdj, Index, Partition};

/// Find connected components of an undirected graph given as CSR adjacency.
/// Returns Partition where item_group[v] = component id (0-based, -1 impossible since all get assigned).
pub fn connected_components(csr: &CsrAdj) -> Partition {
    let nverts = csr.nverts();
    let mut group = vec![-1i32; nverts];
    let mut queue: Vec<Index> = Vec::with_capacity(nverts);
    let mut next_group = 0i32;
    for start in 0..nverts {
        if group[start] >= 0 { continue; }
        group[start] = next_group;
        queue.push(start as Index);
        while let Some(v) = queue.pop() {
            let vi = v as usize;
            let (ns, _) = csr.neighbors(vi);
            for &n in ns {
                let ni = n as usize;
                if group[ni] < 0 {
                    group[ni] = next_group;
                    queue.push(n);
                }
            }
        }
        next_group += 1;
    }
    Partition { item_group: group }
}

/// Split a graph into subgraphs by component. Returns vertex index lists per component.
/// Each component's vertices are listed in ascending order.
pub fn split_by_component(csr: &CsrAdj) -> Vec<Vec<Index>> {
    let part = connected_components(csr);
    let ncomp = part.ngroups();
    let mut comps: Vec<Vec<Index>> = vec![Vec::new(); ncomp];
    for (v, &g) in part.item_group.iter().enumerate() {
        comps[g as usize].push(v as Index);
    }
    comps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency::build_csr_adj;

    #[test]
    fn test_single_component() {
        // Path 0-1-2-3 → one component
        let edges: Vec<[Index; 2]> = vec![[0, 1], [1, 2], [2, 3]];
        let csr = build_csr_adj(4, &edges);
        let part = connected_components(&csr);
        assert_eq!(part.ngroups(), 1);
        for g in &part.item_group { assert_eq!(*g, 0); }
    }

    #[test]
    fn test_two_components() {
        // 0-1-2 and 3-4 → two components
        let edges: Vec<[Index; 2]> = vec![[0, 1], [1, 2], [3, 4]];
        let csr = build_csr_adj(5, &edges);
        let part = connected_components(&csr);
        assert_eq!(part.ngroups(), 2);
        assert_eq!(part.group_of(0), part.group_of(1));
        assert_eq!(part.group_of(1), part.group_of(2));
        assert_eq!(part.group_of(3), part.group_of(4));
        assert_ne!(part.group_of(0), part.group_of(3));
    }

    #[test]
    fn test_isolated_vertices() {
        // 0-1 and 2, 3 isolated → 3 components
        let edges: Vec<[Index; 2]> = vec![[0, 1]];
        let csr = build_csr_adj(4, &edges);
        let comps = split_by_component(&csr);
        assert_eq!(comps.len(), 3);
        // Each isolated vertex is its own component
        let sizes: Vec<usize> = comps.iter().map(|c| c.len()).collect();
        assert!(sizes.contains(&2)); // {0,1}
        assert!(sizes.contains(&1)); // {2}
        assert!(sizes.contains(&1)); // {3}
    }
}
