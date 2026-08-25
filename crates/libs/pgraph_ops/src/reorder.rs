//! Reorder / compact: Partition → IndexGroups, group-aware permutation → RangeGroups.
//! Implements the count → prefix → scatter pattern from FireCore Groups::setGroupMapping().

use pgraph::{Index, IndexGroups, Partition, Permutation, RangeGroups};

/// Convert a Partition to packed IndexGroups (count → prefix → scatter).
/// Each item appears exactly once in its group. Unassigned items (group = -1) are dropped.
pub fn partition_to_index_groups(part: &Partition) -> IndexGroups {
    let ngroups = part.ngroups();
    // 1. Count items per group
    let mut counts = vec![0u32; ngroups];
    for &g in &part.item_group {
        if g >= 0 { counts[g as usize] += 1; }
    }
    // 2. Prefix sum → offsets
    let mut offsets = Vec::with_capacity(ngroups + 1);
    offsets.push(0);
    let mut acc = 0u32;
    for &c in &counts { acc += c; offsets.push(acc); }
    // 3. Scatter items into packed array
    let total = acc as usize;
    let mut items = vec![0 as Index; total];
    let mut cursor = offsets[..ngroups].to_vec();
    for (item, &g) in part.item_group.iter().enumerate() {
        if g < 0 { continue; }
        let gi = g as usize;
        let pos = cursor[gi] as usize;
        cursor[gi] += 1;
        items[pos] = item as Index;
    }
    IndexGroups { offsets, items }
}

/// Build a group-aware permutation that reorders items so each group is contiguous.
/// Returns (Permutation, RangeGroups) where RangeGroups describes the contiguous ranges.
///
/// Items not assigned to any group (group = -1) are placed at the end, after all groups.
pub fn group_aware_permutation(part: &Partition) -> (Permutation, RangeGroups) {
    let nitems = part.nitems();
    let ngroups = part.ngroups();
    let groups = partition_to_index_groups(part);
    // Count unassigned
    let _n_unassigned = part.item_group.iter().filter(|&&g| g < 0).count();
    // Build new2old: group 0 items, group 1 items, ..., unassigned items
    let mut new2old: Vec<Index> = Vec::with_capacity(nitems);
    let mut ranges: Vec<[Index; 2]> = Vec::with_capacity(ngroups);
    let mut start = 0u32;
    for g in 0..ngroups {
        let grp = groups.group(g);
        for &item in grp { new2old.push(item); }
        let end = start + grp.len() as u32;
        ranges.push([start, end]);
        start = end;
    }
    // Append unassigned items
    for (item, &g) in part.item_group.iter().enumerate() {
        if g < 0 { new2old.push(item as Index); }
    }
    assert_eq!(new2old.len(), nitems, "permutation must cover all items");
    let perm = Permutation::from_new2old(new2old);
    (perm, RangeGroups { ranges })
}

/// Apply a permutation to a slice, producing a new Vec. `perm.new2old` maps new index → old index.
pub fn apply_permutation<T: Clone>(perm: &Permutation, data: &[T]) -> Vec<T> {
    assert_eq!(perm.len(), data.len(), "permutation length must match data length");
    perm.new2old.iter().map(|&old| data[old as usize].clone()).collect()
}

/// Apply a permutation to an edge list, remapping both endpoints.
pub fn permute_edges(perm: &Permutation, edges: &[[Index; 2]]) -> Vec<[Index; 2]> {
    edges.iter().map(|&[a, b]| [
        perm.old2new[a as usize],
        perm.old2new[b as usize],
    ]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_to_index_groups() {
        // 3 items: item 0 → group 1, item 1 → group 0, item 2 → group 1
        let mut part = Partition::new(3);
        part.assign(0, 1);
        part.assign(1, 0);
        part.assign(2, 1);
        let groups = partition_to_index_groups(&part);
        assert_eq!(groups.ngroups(), 2);
        // Group 0: {1}
        assert_eq!(groups.group(0), &[1]);
        // Group 1: {0, 2}
        let g1: std::collections::HashSet<Index> = groups.group(1).iter().copied().collect();
        assert_eq!(g1, [0, 2].into_iter().collect());
    }

    #[test]
    fn test_partition_with_unassigned() {
        let mut part = Partition::new(4);
        part.assign(0, 0);
        part.assign(2, 0);
        // items 1, 3 unassigned
        let groups = partition_to_index_groups(&part);
        assert_eq!(groups.ngroups(), 1);
        let g0: std::collections::HashSet<Index> = groups.group(0).iter().copied().collect();
        assert_eq!(g0, [0, 2].into_iter().collect());
    }

    #[test]
    fn test_group_aware_permutation() {
        // 5 items: group 0 = {1, 3}, group 1 = {0, 4}, unassigned = {2}
        let mut part = Partition::new(5);
        part.assign(1, 0);
        part.assign(3, 0);
        part.assign(0, 1);
        part.assign(4, 1);
        // item 2 unassigned
        let (perm, ranges) = group_aware_permutation(&part);
        assert_eq!(perm.len(), 5);
        assert_eq!(ranges.ngroups(), 2);
        // Group 0 should be contiguous: range [0, 2)
        assert_eq!(ranges.group(0), (0, 2));
        // Group 1 should be contiguous: range [2, 4)
        assert_eq!(ranges.group(1), (2, 4));
        // Verify group 0 contains old items {1, 3}
        let g0_items: std::collections::HashSet<Index> = (0..2)
            .map(|i| perm.new2old[i]).collect();
        assert_eq!(g0_items, [1, 3].into_iter().collect());
        // Verify group 1 contains old items {0, 4}
        let g1_items: std::collections::HashSet<Index> = (2..4)
            .map(|i| perm.new2old[i]).collect();
        assert_eq!(g1_items, [0, 4].into_iter().collect());
        // Unassigned item 2 at the end
        assert_eq!(perm.new2old[4], 2);
    }

    #[test]
    fn test_apply_permutation() {
        let data = vec!["a", "b", "c", "d", "e"];
        let perm = Permutation::from_new2old(vec![4, 3, 2, 1, 0]); // reverse
        let result = apply_permutation(&perm, &data);
        assert_eq!(result, vec!["e", "d", "c", "b", "a"]);
    }

    #[test]
    fn test_permute_edges() {
        // Edges: 0-1, 2-3. Permutation: old 0→new 1, old 1→new 0, old 2→new 3, old 3→new 2
        let edges: Vec<[Index; 2]> = vec![[0, 1], [2, 3]];
        let perm = Permutation::from_new2old(vec![1, 0, 3, 2]);
        let new_edges = permute_edges(&perm, &edges);
        assert_eq!(new_edges, vec![[1, 0], [3, 2]]);
    }
}
