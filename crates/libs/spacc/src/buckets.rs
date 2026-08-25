//! Spatial hashing / bucketing. Rebuildable broad-phase acceleration structure.
//!
//! Packed per-cell item lists using the count → prefix → scatter pattern.
//! No allocation during rebuild: `counts` doubles as the per-cell cursor.

use numtypes::Index;

/// Spatial bucket structure: maps items to cells, then provides packed per-cell item lists.
pub struct Buckets {
    pub ncells: usize,
    pub offsets: Vec<u32>, // ncells + 1, offsets[ncells] = total valid items
    pub items:   Vec<u32>, // packed object indices by cell
    counts:      Vec<u32>, // scratch; reused as cursor during scatter
}

impl Buckets {
    /// Allocate for `ncells` cells. No object count needed until build.
    pub fn new(ncells: usize) -> Self {
        Self { ncells, offsets: vec![0; ncells + 1], items: Vec::new(), counts: vec![0; ncells] }
    }

    /// One-shot build from `cell_of_obj`. Items with `cell = -1` are skipped.
    pub fn build(&mut self, cell_of_obj: &[i32]) {
        for c in &mut self.counts { *c = 0; }
        for &c in cell_of_obj {
            if c >= 0 { self.counts[c as usize] += 1; }
        }
        // Prefix sum
        let mut acc = 0u32;
        for i in 0..self.ncells {
            self.offsets[i] = acc;
            acc += self.counts[i];
        }
        self.offsets[self.ncells] = acc;
        // Scatter: reuse counts as cursors
        self.items.resize(acc as usize, 0);
        for c in &mut self.counts { *c = 0; }
        for (obj, &c) in cell_of_obj.iter().enumerate() {
            if c < 0 { continue; }
            let ci = c as usize;
            let pos = (self.offsets[ci] + self.counts[ci]) as usize;
            self.counts[ci] += 1;
            self.items[pos] = obj as u32;
        }
    }

    /// Get all object indices in cell `c`. Empty slice if cell is empty.
    #[inline]
    pub fn cell_objects(&self, c: usize) -> &[Index] {
        let i0 = self.offsets[c] as usize;
        let i1 = self.offsets[c + 1] as usize;
        &self.items[i0..i1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_buckets_basic() {
        // 4 objects in 3 cells: cell 0 = {0, 2}, cell 1 = {1}, cell 2 = {3}
        let cell_of_obj = vec![0, 1, 0, 2];
        let mut buckets = Buckets::new(3);
        buckets.build(&cell_of_obj);
        let c0: HashSet<u32> = buckets.cell_objects(0).iter().copied().collect();
        assert_eq!(c0, [0, 2].into_iter().collect());
        let c1: HashSet<u32> = buckets.cell_objects(1).iter().copied().collect();
        assert_eq!(c1, [1].into_iter().collect());
        let c2: HashSet<u32> = buckets.cell_objects(2).iter().copied().collect();
        assert_eq!(c2, [3].into_iter().collect());
    }

    #[test]
    fn test_buckets_empty_cell() {
        let cell_of_obj = vec![0, 2];
        let mut buckets = Buckets::new(4);
        buckets.build(&cell_of_obj);
        assert!(buckets.cell_objects(1).is_empty());
        assert!(buckets.cell_objects(3).is_empty());
        assert_eq!(buckets.cell_objects(0).len(), 1);
        assert_eq!(buckets.cell_objects(2).len(), 1);
    }

    #[test]
    fn test_buckets_skip_unassigned() {
        let cell_of_obj = vec![0, -1, 0];
        let mut buckets = Buckets::new(2);
        buckets.build(&cell_of_obj);
        let c0: HashSet<u32> = buckets.cell_objects(0).iter().copied().collect();
        assert_eq!(c0, [0, 2].into_iter().collect());
        assert!(buckets.cell_objects(1).is_empty());
    }
}
