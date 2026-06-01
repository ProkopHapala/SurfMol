use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;

pub struct AlignedVec<T, const A: usize> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    _phantom: PhantomData<T>,
}

impl<T, const A: usize> AlignedVec<T, A> {
    pub fn new() -> Self {
        assert!(A.is_power_of_two());
        Self { ptr: NonNull::dangling(), len: 0, cap: 0, _phantom: PhantomData }
    }

    pub fn with_len_fill(len: usize, fill: T) -> Self where T: Copy {
        let mut v = Self::with_capacity(len);
        unsafe {
            let p = v.ptr.as_ptr();
            for i in 0..len { p.add(i).write(fill); }
        }
        v.len = len;
        v
    }

    pub fn with_capacity(cap: usize) -> Self {
        assert!(A.is_power_of_two());
        if cap == 0 { return Self::new(); }
        let size = cap.checked_mul(mem::size_of::<T>()).expect("cap overflow");
        let layout = Layout::from_size_align(size, A.max(mem::align_of::<T>())).expect("layout");
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw as *mut T).expect("alloc failed");
        Self { ptr, len: 0, cap, _phantom: PhantomData }
    }

    #[inline(always)] pub fn len(&self) -> usize { self.len }
    #[inline(always)] pub fn as_slice(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) } }
    #[inline(always)] pub fn as_mut_slice(&mut self) -> &mut [T] { unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) } }

    pub fn resize_fill(&mut self, new_len: usize, fill: T) where T: Copy {
        if new_len > self.cap { *self = Self::with_capacity(new_len); }
        unsafe {
            let p = self.ptr.as_ptr();
            for i in 0..new_len { p.add(i).write(fill); }
        }
        self.len = new_len;
    }
}

impl<T, const A: usize> Drop for AlignedVec<T, A> {
    fn drop(&mut self) {
        if self.cap == 0 { return; }
        let size = self.cap * mem::size_of::<T>();
        let layout = Layout::from_size_align(size, A.max(mem::align_of::<T>())).expect("layout");
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, layout); }
    }
}
