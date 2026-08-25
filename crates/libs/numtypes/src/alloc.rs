//! Aligned vector for HPC arrays. Low-level `unsafe` allocation; safe API upward.
//!
//! Intended for `Copy`/POD-like values. No complicated `Drop` semantics — `resize_fill`
//! and `with_len_fill` require `T: Copy` so overwriting old data is sound.

use std::alloc::{alloc, alloc_zeroed, dealloc, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

pub struct AlignedVec<T, const A: usize> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    _phantom: PhantomData<T>,
}

impl<T, const A: usize> AlignedVec<T, A> {
    /// Empty vector, dangling pointer.
    pub fn new() -> Self {
        assert!(A.is_power_of_two() && A >= mem::align_of::<T>(), "AlignedVec: alignment A must be power-of-two >= align_of::<T>()");
        Self { ptr: NonNull::dangling(), len: 0, cap: 0, _phantom: PhantomData }
    }

    /// Allocate `cap` slots, zero-initialized. Exposed len = 0, so zeroed bytes are not read as `T`.
    /// `unsafe` is used for allocation only.
    pub fn with_capacity(cap: usize) -> Self {
        assert!(A.is_power_of_two() && A >= mem::align_of::<T>(), "AlignedVec: alignment A must be power-of-two >= align_of::<T>()");
        if cap == 0 { return Self::new(); }
        let size = cap.checked_mul(mem::size_of::<T>()).expect("AlignedVec: capacity overflow");
        let layout = Layout::from_size_align(size, A).expect("AlignedVec: invalid layout");
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut T).expect("AlignedVec: allocation failed");
        Self { ptr, len: 0, cap, _phantom: PhantomData }
    }

    /// Allocate `len` slots and fill every slot with `fill`.
    pub fn with_len_fill(len: usize, fill: T) -> Self where T: Copy {
        let mut v = Self::with_capacity(len);
        unsafe {
            let p = v.ptr.as_ptr();
            for i in 0..len { p.add(i).write(fill); }
        }
        v.len = len;
        v
    }

    /// Allocate `len` slots, zero-initialized, and set `len`. Requires `T` to be zero-initializable.
    pub fn with_len(len: usize) -> Self where T: bytemuck::Zeroable {
        assert!(A.is_power_of_two() && A >= mem::align_of::<T>(), "AlignedVec: alignment A must be power-of-two >= align_of::<T>()");
        if len == 0 { return Self::new(); }
        let size = len.checked_mul(mem::size_of::<T>()).expect("AlignedVec: capacity overflow");
        let layout = Layout::from_size_align(size, A).expect("AlignedVec: invalid layout");
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut T).expect("AlignedVec: allocation failed");
        Self { ptr, len, cap: len, _phantom: PhantomData }
    }

    #[inline(always)] pub fn len(&self) -> usize { self.len }
    #[inline(always)] pub fn capacity(&self) -> usize { self.cap }
    #[inline(always)] pub fn is_empty(&self) -> bool { self.len == 0 }
    #[inline(always)] pub fn as_ptr(&self) -> *const T { self.ptr.as_ptr() }
    #[inline(always)] pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr.as_ptr() }
    #[inline(always)] pub fn as_slice(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) } }
    #[inline(always)] pub fn as_mut_slice(&mut self) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) } }

    /// Append `val` to the end. Grows if needed. O(1) amortized, preserves alignment.
    pub fn push(&mut self, val: T) where T: Copy {
        if self.len == self.cap { self.grow(std::cmp::max(1, self.cap * 2)); }
        unsafe { self.ptr.as_ptr().add(self.len).write(val); }
        self.len += 1;
    }

    /// Clear length to 0; does not deallocate. Values of type `T` are not dropped —
    /// `AlignedVec` is intended for `Copy`/POD types.
    pub fn clear(&mut self) { self.len = 0; }

    /// Resize to `new_len`, filling all slots with `fill` (not just new slots).
    /// This matches the existing SurfMol usage pattern where AlignedVec is used to
    /// initialize force/topology buffers to a known sentinel.
    pub fn resize_fill(&mut self, new_len: usize, fill: T) where T: Copy {
        if new_len > self.cap { self.grow(new_len); }
        unsafe {
            let p = self.ptr.as_ptr();
            for i in 0..new_len { p.add(i).write(fill); }
        }
        self.len = new_len;
    }

    fn grow(&mut self, new_cap: usize) where T: Copy {
        assert!(new_cap >= self.len, "AlignedVec::grow: new capacity must fit current length");
        let new_size = new_cap.checked_mul(mem::size_of::<T>()).expect("AlignedVec: grow overflow");
        let new_layout = Layout::from_size_align(new_size, A).expect("AlignedVec: invalid layout");
        let new_raw = unsafe { alloc(new_layout) };
        let new_ptr = NonNull::new(new_raw as *mut T).expect("AlignedVec: allocation failed");
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.len);
        }
        if self.cap > 0 {
            let old_size = self.cap * mem::size_of::<T>();
            let old_layout = Layout::from_size_align(old_size, A).expect("AlignedVec: invalid layout");
            unsafe { dealloc(self.ptr.as_ptr() as *mut u8, old_layout); }
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
    }
}

impl<T, const A: usize> Deref for AlignedVec<T, A> {
    type Target = [T];
    #[inline(always)] fn deref(&self) -> &[T] { self.as_slice() }
}

impl<T, const A: usize> DerefMut for AlignedVec<T, A> {
    #[inline(always)] fn deref_mut(&mut self) -> &mut [T] { self.as_mut_slice() }
}

impl<T, const A: usize> Drop for AlignedVec<T, A> {
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        unsafe {
            std::ptr::drop_in_place(self.as_mut_slice());
        }
        let size = self.cap * mem::size_of::<T>();
        let layout = Layout::from_size_align(size, A).expect("AlignedVec: invalid layout");
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout); }
    }
}

impl<T, const A: usize> fmt::Debug for AlignedVec<T, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVec").field("len", &self.len).field("cap", &self.cap).field("align", &A).field("ptr", &self.ptr).finish()
    }
}

impl<T: Copy, const A: usize> From<&[T]> for AlignedVec<T, A> {
    fn from(src: &[T]) -> Self {
        let mut v = Self::with_capacity(src.len());
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), v.ptr.as_ptr(), src.len()); }
        v.len = src.len();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_vec_basic() {
        let mut v = AlignedVec::<f64, 64>::with_len_fill(4, 1.0);
        assert_eq!(v.len(), 4);
        assert!(v.as_ptr() as usize % 64 == 0);
        assert_eq!(v.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
        v[0] = 2.0;
        assert_eq!(v[0], 2.0);
    }

    #[test]
    fn aligned_vec_resize_fill() {
        let mut v = AlignedVec::<i32, 64>::new();
        v.resize_fill(3, 7);
        assert_eq!(v.as_slice(), &[7, 7, 7]);
        v.resize_fill(5, 9);
        assert_eq!(v.as_slice(), &[9, 9, 9, 9, 9]);
    }

    #[test]
    fn aligned_vec_from_slice() {
        let v = AlignedVec::<i32, 64>::from(&[1, 2, 3][..]);
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }
}
