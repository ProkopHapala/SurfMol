//! Spatial hashing / bucketing. Rebuildable broad-phase acceleration structure.
//! Ported from FireCore/SSE Buckets.h — count → prefix → scatter pattern.

/// Spatial bucket structure: maps items to spatial cells, then provides
/// packed per-cell item lists for O(1) "all items in cell c" queries.
///
/// Build by providing a cell index per item, then call `build` to pack.
/// Same pattern as FireCore `Buckets` and `NBFF::initBBsFromGroups()`.
pub struct Buckets {
    pub ncells: usize,
    pub cell_ns:  Vec<i32>,   // count per cell
    pub cell_i0s: Vec<i32>,   // start index per cell (prefix sum)
    pub cell2obj: Vec<i32>,   // object indices packed by cell
    pub nobjs: i32,
}

impl Buckets {
    /// Allocate for `ncells` cells and `nobjs` objects. Does not fill.
    pub fn new(ncells: usize, nobjs: usize) -> Self {
        Self {
            ncells,
            cell_ns:  vec![0; ncells],
            cell_i0s: vec![0; ncells],
            cell2obj: vec![-1; nobjs],
            nobjs: nobjs as i32,
        }
    }

    /// Count items per cell. Call this first, then `update_offsets`, then `scatter`.
    pub fn count(&mut self, cell_of_obj: &[i32]) {
        for c in &mut self.cell_ns { *c = 0; }
        for &c in cell_of_obj {
            if c >= 0 { self.cell_ns[c as usize] += 1; }
        }
    }

    /// Compute prefix sums to get cell start offsets. Call after `count`.
    pub fn update_offsets(&mut self) {
        let mut acc = 0i32;
        for i in 0..self.ncells {
            self.cell_i0s[i] = acc;
            acc += self.cell_ns[i];
        }
    }

    /// Scatter object indices into packed cells. Call after `update_offsets`.
    /// Uses a running cursor per cell (copy of cell_i0s).
    pub fn scatter(&mut self, cell_of_obj: &[i32]) {
        let mut cursor = self.cell_i0s.clone();
        for (obj, &c) in cell_of_obj.iter().enumerate() {
            if c < 0 { continue; }
            let pos = cursor[c as usize] as usize;
            cursor[c as usize] += 1;
            self.cell2obj[pos] = obj as i32;
        }
    }

    /// One-shot build: count → prefix → scatter. Convenience over the three steps.
    pub fn build(&mut self, cell_of_obj: &[i32]) {
        self.count(cell_of_obj);
        self.update_offsets();
        self.scatter(cell_of_obj);
    }

    /// Get all object indices in cell `c`. Empty slice if cell is empty.
    #[inline]
    pub fn cell_objects(&self, c: usize) -> &[i32] {
        let i0 = self.cell_i0s[c] as usize;
        let n = self.cell_ns[c] as usize;
        &self.cell2obj[i0..i0 + n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buckets_basic() {
        // 4 objects in 3 cells: cell 0 = {0, 2}, cell 1 = {1}, cell 2 = {3}
        let cell_of_obj = vec![0, 1, 0, 2];
        let mut buckets = Buckets::new(3, 4);
        buckets.build(&cell_of_obj);
        // Cell 0: objects 0 and 2
        let c0: std::collections::HashSet<i32> = buckets.cell_objects(0).iter().copied().collect();
        assert_eq!(c0, [0, 2].into_iter().collect());
        // Cell 1: object 1
        let c1: std::collections::HashSet<i32> = buckets.cell_objects(1).iter().copied().collect();
        assert_eq!(c1, [1].into_iter().collect());
        // Cell 2: object 3
        let c2: std::collections::HashSet<i32> = buckets.cell_objects(2).iter().copied().collect();
        assert_eq!(c2, [3].into_iter().collect());
    }

    #[test]
    fn test_buckets_empty_cell() {
        // 2 objects, 4 cells: cell 0 = {0}, cell 2 = {1}, cells 1,3 empty
        let cell_of_obj = vec![0, 2];
        let mut buckets = Buckets::new(4, 2);
        buckets.build(&cell_of_obj);
        assert!(buckets.cell_objects(1).is_empty());
        assert!(buckets.cell_objects(3).is_empty());
        assert_eq!(buckets.cell_objects(0).len(), 1);
        assert_eq!(buckets.cell_objects(2).len(), 1);
    }

    #[test]
    fn test_buckets_skip_unassigned() {
        // Object with cell = -1 should be skipped
        let cell_of_obj = vec![0, -1, 0];
        let mut buckets = Buckets::new(2, 3);
        buckets.build(&cell_of_obj);
        let c0: std::collections::HashSet<i32> = buckets.cell_objects(0).iter().copied().collect();
        assert_eq!(c0, [0, 2].into_iter().collect());
        assert!(buckets.cell_objects(1).is_empty());
    }
}
