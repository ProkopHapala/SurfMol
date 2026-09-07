//! OpenCL GPU harness for UFF (FireCore `UFF.cl`).
//!
//! Ports FireCore `cpp/common_resources/cl/UFF.cl:evalBondsAndHNeigh_UFF` etc.
//! CPU reference (authoritative for correctness): `molff::uff::Uff`.
//!
//! UFF.cl is **self-contained** (defines its own `cl_Mat3`, `R2SAFE`, `EXCL_MAX`).
//! Load it as a standalone OpenCL program.

use ocl::{ProQue, Buffer, flags};

const UFF_SRC: &str = include_str!("../../../../opencl/UFF.cl");

/// GPU UFF harness. Wraps a single `ProQue` compiled from `opencl/UFF.cl`.
pub struct UffOcl { pro_que: ProQue }

impl UffOcl {
    /// Compile `opencl/UFF.cl` on the first available NVIDIA GPU.
    pub fn new() -> ocl::Result<Self> {
        Ok(Self { pro_que: crate::nvidia_proque(UFF_SRC)? })
    }

    pub fn pro_que(&self) -> &ProQue { &self.pro_que }

    /// One-system, bond-only force evaluation. Returns `fapos` `{fx,fy,fz,E}` per atom.
    ///
    /// Inputs are 1-system host arrays. `neighs`/`neighCell`/`neighBs` are per-atom `int4`
    /// padded with `-1`; `bonAtoms[ib]={ia,ja}` and `bonParams[ib]={K,l0}`; `req[ia]={R0,E0,Q,_}`.
    ///
    /// Work size: 1-D `natoms` for `clear_fapos_UFF`, 2-D `(natoms,1)` for `evalBondsAndHNeigh_UFF`.
    pub fn eval_bonds(
        &self,
        apos: &[[f32; 4]],
        bon_atoms: &[[i32; 2]],
        bon_params: &[[f32; 2]],
        neighs: &[[i32; 4]],
        neigh_cell: &[[i32; 4]],
        neigh_bs: &[[i32; 4]],
        pbc_shifts: &[[f32; 4]],
        req: &[[f32; 4]],
        rdamp: f32,
        fmax_nb: f32,
        b_sub_nb: i32,
    ) -> ocl::Result<Vec<[f32; 4]>> {
        let natoms = apos.len() as i32;
        let npbc = pbc_shifts.len() as i32;

        let apos_buf = self.upload_f4(apos)?;
        let bon_atoms_buf = self.upload_i2(bon_atoms)?;
        let bon_params_buf = self.upload_f2(bon_params)?;
        let neighs_buf = self.upload_i4(neighs)?;
        let neigh_cell_buf = self.upload_i4(neigh_cell)?;
        let neigh_bs_buf = self.upload_i4(neigh_bs)?;
        let pbc_shifts_buf = self.upload_f4(pbc_shifts)?;
        let req_buf = self.upload_f4(req)?;

        let fapos = self.zeros_f4(natoms as usize)?;
        let hneigh = self.zeros_f4(natoms as usize * 4)?;
        let fint = self.zeros_f4(1)?; // bond kernel has fint arg but does not write it

        let clear = self.pro_que.kernel_builder("clear_fapos_UFF")
            .arg(natoms).arg(&fapos)
            .global_work_size(natoms as usize)
            .build()?;
        unsafe { clear.enq()?; }

        let k = self.pro_que.kernel_builder("evalBondsAndHNeigh_UFF")
            .arg(natoms)
            .arg(npbc)
            .arg(0i32) // i0bon
            .arg(b_sub_nb)
            .arg(rdamp)
            .arg(fmax_nb)
            .arg(&apos_buf)
            .arg(&fapos)
            .arg(&neighs_buf)
            .arg(&neigh_cell_buf)
            .arg(&pbc_shifts_buf)
            .arg(&neigh_bs_buf)
            .arg(&bon_params_buf)
            .arg(&req_buf)
            .arg(&bon_atoms_buf)
            .arg(&hneigh)
            .arg(&fint)
            .global_work_size([natoms as usize, 1])
            .build()?;
        unsafe { k.enq()?; }

        let mut out = vec![0.0f32; natoms as usize * 4];
        fapos.read(&mut out).enq()?;
        Ok((0..natoms as usize).map(|i| [out[i*4], out[i*4+1], out[i*4+2], out[i*4+3]]).collect())
    }

    fn queue(&self) -> ocl::Queue { self.pro_que.queue().clone() }

    fn upload_f4(&self, v: &[[f32; 4]]) -> ocl::Result<Buffer<f32>> {
        let flat: Vec<f32> = v.iter().flat_map(|&a| a).collect();
        Buffer::builder().queue(self.queue()).flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR).len(flat.len()).copy_host_slice(&flat).build()
    }
    fn upload_f2(&self, v: &[[f32; 2]]) -> ocl::Result<Buffer<f32>> {
        let flat: Vec<f32> = v.iter().flat_map(|&a| a).collect();
        Buffer::builder().queue(self.queue()).flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR).len(flat.len()).copy_host_slice(&flat).build()
    }
    fn upload_i4(&self, v: &[[i32; 4]]) -> ocl::Result<Buffer<i32>> {
        let flat: Vec<i32> = v.iter().flat_map(|&a| a).collect();
        Buffer::builder().queue(self.queue()).flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR).len(flat.len()).copy_host_slice(&flat).build()
    }
    fn upload_i2(&self, v: &[[i32; 2]]) -> ocl::Result<Buffer<i32>> {
        let flat: Vec<i32> = v.iter().flat_map(|&a| a).collect();
        Buffer::builder().queue(self.queue()).flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR).len(flat.len()).copy_host_slice(&flat).build()
    }
    fn zeros_f4(&self, n: usize) -> ocl::Result<Buffer<f32>> {
        let zeros = vec![0.0f32; n * 4];
        Buffer::builder().queue(self.queue()).flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR).len(zeros.len()).copy_host_slice(&zeros).build()
    }
}
