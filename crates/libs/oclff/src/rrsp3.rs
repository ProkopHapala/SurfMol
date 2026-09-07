//! RRsp3 — OpenCL GPU harness for the cluster-sorted rigid-atom port forcefield.
//!
//! Port of FireCore `pyBall/RigidAtomFF/RRsp3/RRsp3.py`. Manages OpenCL context,
//! persistent buffers, kernel dispatch for the cluster-sorted layout (Axis 4b):
//! one workgroup per molecule, nodes first, ghost atoms in local memory.
//!
//! Kernel source: `opencl/RRsp3.cl` (copied from FireCore, identical physics).
//! CPU reference (authoritative): `molff::raff`.

use ocl::{self, Buffer, Context, Device, Kernel, Platform, Program, Queue, flags, flags::MemFlags};
use ocl::enums::{DeviceInfo, DeviceInfoResult};
use ocl::builders::ProgramBuilder;

/// OpenCL kernel source (embedded at compile time from `opencl/RRsp3.cl`).
const RRSP3_SRC: &str = include_str!("../../../../opencl/RRsp3.cl");

// ------------------------------------------------------------------
//  Config
// ------------------------------------------------------------------

/// Which port-force kernel variant to use (matches RRsp3.py `port_kernel` arg).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PortKernel {
    /// Massfull XPBD with physical rotational inertia. `compute_ports_cluster_rigid`.
    Current,
    /// Massfull, original (no rot_mass_scale). `compute_ports_cluster_rigid_orig`.
    Orig,
    /// Massless Newton-Raphson in omega-space. `compute_ports_cluster_rigid_substep_optimized`.
    Substep,
    /// Massless polar/Kabsch decomposition. `compute_ports_cluster_rigid_shapematch`.
    Shapematch,
    /// Massless Horn quaternion eigen (two-pass). `compute_optimal_rotation_eigen` + `compute_ports_cluster_rigid_eigen_tips`.
    Eigen,
}

impl PortKernel {
    pub fn is_massless(&self) -> bool {
        matches!(self, PortKernel::Substep | PortKernel::Shapematch | PortKernel::Eigen)
    }
    pub fn from_str(s: &str) -> Self {
        match s {
            "current" | "rigid" => PortKernel::Current,
            "orig" | "original" | "orid" => PortKernel::Orig,
            "substep" | "substep_optimized" => PortKernel::Substep,
            "shapematch" | "shape_match" => PortKernel::Shapematch,
            "eigen" | "q_eigen" => PortKernel::Eigen,
            _ => panic!("PortKernel::from_str: unknown '{s}'; use current/orig/substep/shapematch/eigen"),
        }
    }
}

/// Step configuration (matches RRsp3.py step_cluster / step_dynamics kwargs).
#[derive(Copy, Clone, Debug)]
pub struct StepConfig {
    pub dt: f32,
    pub k_coll: f32,
    pub relaxation: f32,
    pub bbox_margin: f32,
    pub momentum_beta: f32,      // heavy-ball momentum (0 = disabled)
    pub rot_mass_scale: f32,     // for 'current' kernel
    pub n_rot_substeps: i32,     // for 'substep' kernel
    pub rot_eps: f32,            // for 'substep' kernel
    pub theta_max: f32,          // for 'substep' kernel
    pub damp: f32,               // for dynamics (velocity damping)
}

impl Default for StepConfig {
    fn default() -> Self {
        Self { dt: 0.1, k_coll: 50.0, relaxation: 0.5, bbox_margin: 0.5,
            momentum_beta: 0.0, rot_mass_scale: 1.0, n_rot_substeps: 5,
            rot_eps: 0.0, theta_max: 0.0, damp: 1.0 }
    }
}

// ------------------------------------------------------------------
//  RRsp3 harness
// ------------------------------------------------------------------

/// OpenCL GPU harness for the RRsp3 cluster-sorted rigid-atom forcefield.
///
/// Manages persistent buffers and kernel dispatch. One instance per system size
/// (natoms must be a multiple of group_size). Port of `RRsp3.py::RRsp3`.
pub struct RRsp3 {
    pub natoms: usize,
    pub group_size: usize,
    pub num_groups: usize,
    pub max_ghosts: usize,
    pub nnode_per_group: i32,
    pub nnode_tot: usize,

    queue: Queue,
    program: Program,
    device_name: String,
    // Persistent state buffers (allocated once)
    cl_pos: Buffer<f32>,       // float4 [natoms]
    cl_quat: Buffer<f32>,      // float4 [natoms]
    cl_radius: Buffer<f32>,    // float  [natoms]
    cl_neighs: Buffer<i32>,    // int4   [natoms]
    cl_excl1: Buffer<i32>,     // int4   [natoms]
    cl_excl2: Buffer<i32>,     // int4   [natoms]
    cl_fixmask: Buffer<i32>,   // int    [natoms]

    cl_bboxes_min: Buffer<f32>, // float4 [num_groups]
    cl_bboxes_max: Buffer<f32>, // float4 [num_groups]
    cl_ghost_indices: Buffer<i32>, // int [num_groups * max_ghosts]
    cl_ghost_counts: Buffer<i32>,  // int [num_groups]
    cl_neighs_local: Buffer<i32>,  // int4 [natoms]
    cl_excl1_local: Buffer<i32>,   // int4 [natoms]
    cl_excl2_local: Buffer<i32>,   // int4 [natoms]

    cl_dpos_coll: Buffer<f32>,     // float4 [natoms]
    cl_dpos_mom: Buffer<f32>,      // float4 [natoms] (heavy-ball momentum)
    cl_dquat_mom: Buffer<f32>,     // float4 [natoms]
    cl_vel: Buffer<f32>,           // float4 [natoms] (dynamics)
    cl_omega: Buffer<f32>,         // float4 [natoms]
    cl_pos_prev: Buffer<f32>,      // float4 [natoms]
    cl_quat_prev: Buffer<f32>,     // float4 [natoms]

    // Node-only buffers (allocated lazily when nnode_per_group is set)
    cl_dpos_node: Option<Buffer<f32>>,    // float4 [nnode_tot]
    cl_drot_node: Option<Buffer<f32>>,    // float4 [nnode_tot]
    cl_dpos_neigh: Option<Buffer<f32>>,   // float4 [nnode_tot * 4]
    cl_tips: Option<Buffer<f32>>,         // float4 [nnode_tot * 4]
    cl_rev_slot: Option<Buffer<i32>>,     // int   [nnode_tot * 4]
    cl_port_local: Option<Buffer<f32>>,   // float4 [nnode_tot * 4]
    cl_kflat: Option<Buffer<f32>>,        // float [nnode_tot * 4]
    cl_bk_slots: Option<Buffer<i32>>,     // int4  [natoms]

    tips_valid: bool,
}

impl RRsp3 {
    /// Create a new RRsp3 harness. `natoms` must be a multiple of `group_size`.
    /// `prefer_gpu` selects a GPU device (prefer NVIDIA); falls back to any device.
    /// Kernel source is embedded via `include_str!("opencl/RRsp3.cl")`.
    pub fn new(natoms: usize, group_size: usize, max_ghosts: usize, prefer_gpu: bool) -> ocl::Result<Self> {
        assert!(natoms % group_size == 0, "RRsp3::new: natoms={natoms} must be multiple of group_size={group_size}");
        assert!(group_size.is_power_of_two(), "RRsp3::new: group_size={group_size} must be power of two (kernel uses lid & (GROUP_SIZE-1))");
        let num_groups = natoms / group_size;

        // Select platform + device (prefer NVIDIA GPU across all platforms)
        let (platform, device) = if prefer_gpu {
            // Search all platforms for an NVIDIA GPU device
            let mut found = None;
            for p in Platform::list() {
                if let Ok(gpus) = Device::list(&p, Some(flags::DEVICE_TYPE_GPU)) {
                    for d in gpus {
                        if let Ok(DeviceInfoResult::Vendor(v)) = d.info(DeviceInfo::Vendor) {
                            if v.contains("NVIDIA") { found = Some((p, d)); break; }
                        }
                    }
                }
                if found.is_some() { break; }
            }
            // Fallback: any GPU on any platform, then any device on default platform
            if let Some(pd) = found { pd }
            else {
                let dp = Platform::default();
                let d = Device::list(&dp, Some(flags::DEVICE_TYPE_GPU))
                    .ok().and_then(|g| g.into_iter().next())
                    .unwrap_or_else(|| Device::first(&dp).expect("RRsp3::new: no OpenCL device"));
                (dp, d)
            }
        } else {
            let dp = Platform::default();
            (dp, Device::first(&dp)?)
        };
        let device_name = device.name()?;

        let context = Context::builder().platform(platform).devices(device.clone()).build()?;
        let queue = Queue::new(&context, device.clone(), None)?;

        let program = ProgramBuilder::new()
            .devices(device.clone()).src(RRSP3_SRC)
            .cmplr_def("GROUP_SIZE", group_size as i32)
            .cmplr_def("MAX_GHOSTS", max_ghosts as i32)
            .build(&context)?;

        let mk_fbuf = |len: usize, fl: MemFlags| -> ocl::Result<Buffer<f32>> {
            Buffer::<f32>::builder().queue(queue.clone()).flags(fl).len(len).build()
        };
        let mk_ibuf = |len: usize, fl: MemFlags| -> ocl::Result<Buffer<i32>> {
            Buffer::<i32>::builder().queue(queue.clone()).flags(fl).len(len).build()
        };

        let n4 = natoms * 4;
        let cl_pos = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_quat = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_radius = mk_fbuf(natoms, flags::MEM_READ_ONLY)?;
        let cl_neighs = mk_ibuf(n4, flags::MEM_READ_ONLY)?;
        let cl_excl1 = mk_ibuf(n4, flags::MEM_READ_ONLY)?;
        let cl_excl2 = mk_ibuf(n4, flags::MEM_READ_ONLY)?;
        let cl_fixmask = mk_ibuf(natoms, flags::MEM_READ_ONLY)?;

        let ng4 = num_groups * 4;
        let cl_bboxes_min = mk_fbuf(ng4, flags::MEM_READ_WRITE)?;
        let cl_bboxes_max = mk_fbuf(ng4, flags::MEM_READ_WRITE)?;
        let cl_ghost_indices = mk_ibuf(num_groups * max_ghosts, flags::MEM_READ_WRITE)?;
        let cl_ghost_counts = mk_ibuf(num_groups, flags::MEM_READ_WRITE)?;
        let cl_neighs_local = mk_ibuf(n4, flags::MEM_READ_WRITE)?;
        let cl_excl1_local = mk_ibuf(n4, flags::MEM_READ_WRITE)?;
        let cl_excl2_local = mk_ibuf(n4, flags::MEM_READ_WRITE)?;

        let cl_dpos_coll = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_dpos_mom = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_dquat_mom = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_vel = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_omega = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_pos_prev = mk_fbuf(n4, flags::MEM_READ_WRITE)?;
        let cl_quat_prev = mk_fbuf(n4, flags::MEM_READ_WRITE)?;

        let zeros_f = vec![0.0f32; n4];
        cl_dpos_mom.write(&zeros_f[..]).enq()?;
        cl_dquat_mom.write(&zeros_f[..]).enq()?;
        cl_vel.write(&zeros_f[..]).enq()?;
        cl_omega.write(&zeros_f[..]).enq()?;
        let zeros_i = vec![0i32; natoms];
        cl_fixmask.write(&zeros_i[..]).enq()?;

        Ok(Self {
            natoms, group_size, num_groups, max_ghosts, nnode_per_group: 0, nnode_tot: 0,
            queue, program, device_name,
            cl_pos, cl_quat, cl_radius, cl_neighs, cl_excl1, cl_excl2, cl_fixmask,
            cl_bboxes_min, cl_bboxes_max, cl_ghost_indices, cl_ghost_counts,
            cl_neighs_local, cl_excl1_local, cl_excl2_local,
            cl_dpos_coll, cl_dpos_mom, cl_dquat_mom, cl_vel, cl_omega, cl_pos_prev, cl_quat_prev,
            cl_dpos_node: None, cl_drot_node: None, cl_dpos_neigh: None,
            cl_tips: None, cl_rev_slot: None, cl_port_local: None, cl_kflat: None, cl_bk_slots: None,
            tips_valid: false,
        })
    }

    pub fn device_name(&self) -> &str { &self.device_name }

    // ------------------------------------------------------------------
    //  Node-only buffer allocation (lazy, sized by nnode_per_group)
    // ------------------------------------------------------------------

    fn ensure_node_buffers(&mut self, nnode_per_group: i32) -> ocl::Result<()> {
        assert!(nnode_per_group >= 0 && nnode_per_group as usize <= self.group_size,
            "ensure_node_buffers: nnode_per_group={nnode_per_group} out of range [0,{}]", self.group_size);
        let nnode_tot = self.num_groups * nnode_per_group as usize;
        if self.nnode_tot == nnode_tot && self.cl_dpos_node.is_some() { return Ok(()); }
        self.nnode_per_group = nnode_per_group;
        self.nnode_tot = nnode_tot;
        let q = self.queue.clone();
        let mk_fbuf = |len: usize, fl: MemFlags| -> ocl::Result<Buffer<f32>> {
            Buffer::<f32>::builder().queue(q.clone()).flags(fl).len(len).build()
        };
        let mk_ibuf = |len: usize, fl: MemFlags| -> ocl::Result<Buffer<i32>> {
            Buffer::<i32>::builder().queue(q.clone()).flags(fl).len(len).build()
        };
        self.cl_dpos_node = Some(mk_fbuf(nnode_tot * 4, flags::MEM_READ_WRITE)?);
        self.cl_drot_node = Some(mk_fbuf(nnode_tot * 4, flags::MEM_READ_WRITE)?);
        self.cl_dpos_neigh = Some(mk_fbuf(nnode_tot * 4 * 4, flags::MEM_READ_WRITE)?);
        self.cl_tips = Some(mk_fbuf(nnode_tot * 4 * 4, flags::MEM_READ_WRITE)?);
        self.cl_rev_slot = Some(mk_ibuf(nnode_tot * 4, flags::MEM_READ_ONLY)?);
        self.cl_port_local = Some(mk_fbuf(nnode_tot * 4 * 4, flags::MEM_READ_ONLY)?);
        self.cl_kflat = Some(mk_fbuf(nnode_tot * 4, flags::MEM_READ_ONLY)?);
        self.cl_bk_slots = Some(mk_ibuf(self.natoms * 4, flags::MEM_READ_ONLY)?);
        self.tips_valid = false;
        Ok(())
    }

    // ------------------------------------------------------------------
    //  Upload
    // ------------------------------------------------------------------

    /// Upload positions (xyz) + inverse masses. Padding atoms (invM<=0) get NaN pos.
    pub fn upload_state(&mut self, pos3: &[[f32; 3]], inv_mass: &[f32], quat: Option<&[[f32; 4]]>) -> ocl::Result<()> {
        assert!(pos3.len() == self.natoms, "upload_state: pos3.len()={} != natoms={}", pos3.len(), self.natoms);
        assert!(inv_mass.len() == self.natoms, "upload_state: inv_mass.len()={} != natoms={}", inv_mass.len(), self.natoms);
        let mut pos4 = vec![0.0f32; self.natoms * 4];
        for i in 0..self.natoms {
            pos4[i * 4 + 0] = pos3[i][0];
            pos4[i * 4 + 1] = pos3[i][1];
            pos4[i * 4 + 2] = pos3[i][2];
            pos4[i * 4 + 3] = inv_mass[i];
            if inv_mass[i] <= 1e-12 { pos4[i * 4 + 0] = f32::NAN; pos4[i * 4 + 1] = f32::NAN; pos4[i * 4 + 2] = f32::NAN; }
        }
        self.cl_pos.write(&pos4[..]).enq()?;
        let mut quat4 = vec![0.0f32; self.natoms * 4];
        if let Some(q) = quat {
            assert!(q.len() == self.natoms, "upload_state: quat.len()={} != natoms={}", q.len(), self.natoms);
            for i in 0..self.natoms { quat4[i * 4..i * 4 + 4].copy_from_slice(&q[i]); }
        } else { for i in 0..self.natoms { quat4[i * 4 + 3] = 1.0; } }
        self.cl_quat.write(&quat4[..]).enq()?;
        self.tips_valid = false;
        Ok(())
    }

    pub fn upload_radius(&mut self, radius: &[f32]) -> ocl::Result<()> {
        assert!(radius.len() == self.natoms, "upload_radius: len={} != natoms={}", radius.len(), self.natoms);
        self.cl_radius.write(&radius[..]).enq()
    }

    pub fn upload_fixmask(&mut self, fixmask: &[i32]) -> ocl::Result<()> {
        assert!(fixmask.len() == self.natoms, "upload_fixmask: len={} != natoms={}", fixmask.len(), self.natoms);
        self.cl_fixmask.write(&fixmask[..]).enq()
    }

    pub fn upload_neighs_and_exclusions(&mut self, neighs: &[i32], excl1: &[i32], excl2: &[i32]) -> ocl::Result<()> {
        assert!(neighs.len() == self.natoms * 4, "upload_neighs: len={} != natoms*4", neighs.len());
        assert!(excl1.len() == self.natoms * 4, "upload_excl1: len={} != natoms*4", excl1.len());
        assert!(excl2.len() == self.natoms * 4, "upload_excl2: len={} != natoms*4", excl2.len());
        self.cl_neighs.write(&neighs[..]).enq()?;
        self.cl_excl1.write(&excl1[..]).enq()?;
        self.cl_excl2.write(&excl2[..]).enq()
    }

    /// Upload cluster port geometry + stiffness (node-only, packed per nnode_per_group).
    pub fn upload_cluster_ports(&mut self, port_local_atoms: &[f32], k_atoms: &[f32], nnode_per_group: i32) -> ocl::Result<()> {
        self.ensure_node_buffers(nnode_per_group)?;
        assert!(port_local_atoms.len() == self.natoms * 4 * 4, "upload_cluster_ports: port_local len={}", port_local_atoms.len());
        assert!(k_atoms.len() == self.natoms * 4, "upload_cluster_ports: k_atoms len={}", k_atoms.len());
        let mut pl = vec![0.0f32; self.nnode_tot * 4 * 4];
        let mut kk = vec![0.0f32; self.nnode_tot * 4];
        for ig in 0..self.num_groups {
            let abase = ig * self.group_size;
            let inode_base = ig * nnode_per_group as usize;
            for il in 0..nnode_per_group as usize {
                let ia = abase + il;
                let inode = inode_base + il;
                for k in 0..4 {
                    let src = (ia * 4 + k) * 4;
                    let dst = (inode * 4 + k) * 4;
                    pl[dst..dst + 4].copy_from_slice(&port_local_atoms[src..src + 4]);
                    kk[inode * 4 + k] = k_atoms[ia * 4 + k];
                }
            }
        }
        self.cl_port_local.as_ref().unwrap().write(&pl[..]).enq()?;
        self.cl_kflat.as_ref().unwrap().write(&kk[..]).enq()
    }

    pub fn upload_bk_slots(&mut self, bk_slots: &[i32]) -> ocl::Result<()> {
        assert!(bk_slots.len() == self.natoms * 4, "upload_bk_slots: len={} != natoms*4", bk_slots.len());
        if self.cl_bk_slots.is_none() { panic!("upload_bk_slots: call upload_cluster_ports first (allocates bk_slots buffer)"); }
        self.cl_bk_slots.as_ref().unwrap().write(&bk_slots[..]).enq()
    }

    pub fn upload_rev_slot(&mut self, rev_slot: &[i32], nnode_per_group: i32) -> ocl::Result<()> {
        self.ensure_node_buffers(nnode_per_group)?;
        assert!(rev_slot.len() == self.nnode_tot * 4, "upload_rev_slot: len={} != nnode_tot*4={}", rev_slot.len(), self.nnode_tot * 4);
        self.cl_rev_slot.as_ref().unwrap().write(&rev_slot[..]).enq()
    }

    pub fn reset_momentum(&mut self) -> ocl::Result<()> {
        let zeros = vec![0.0f32; self.natoms * 4];
        self.cl_dpos_mom.write(&zeros[..]).enq()?;
        self.cl_dquat_mom.write(&zeros[..]).enq()
    }

    pub fn reset_dynamics(&mut self) -> ocl::Result<()> {
        let zeros = vec![0.0f32; self.natoms * 4];
        self.cl_vel.write(&zeros[..]).enq()?;
        self.cl_omega.write(&zeros[..]).enq()
    }

    // ------------------------------------------------------------------
    //  Download
    // ------------------------------------------------------------------

    pub fn download_pos(&self) -> ocl::Result<Vec<[f32; 4]>> {
        let mut out = vec![0.0f32; self.natoms * 4];
        self.cl_pos.read(&mut out[..]).enq()?;
        Ok((0..self.natoms).map(|i| [out[i*4], out[i*4+1], out[i*4+2], out[i*4+3]]).collect())
    }

    pub fn download_quat(&self) -> ocl::Result<Vec<[f32; 4]>> {
        let mut out = vec![0.0f32; self.natoms * 4];
        self.cl_quat.read(&mut out[..]).enq()?;
        Ok((0..self.natoms).map(|i| [out[i*4], out[i*4+1], out[i*4+2], out[i*4+3]]).collect())
    }

    pub fn download_pos_quat(&self) -> ocl::Result<(Vec<[f32; 4]>, Vec<[f32; 4]>)> {
        Ok((self.download_pos()?, self.download_quat()?))
    }

    pub fn download_ghost_counts(&self) -> ocl::Result<Vec<i32>> {
        let mut out = vec![0i32; self.num_groups];
        self.cl_ghost_counts.read(&mut out[..]).enq()?;
        Ok(out)
    }

    pub fn download_ghost_indices(&self) -> ocl::Result<Vec<i32>> {
        let mut out = vec![0i32; self.num_groups * self.max_ghosts];
        self.cl_ghost_indices.read(&mut out[..]).enq()?;
        Ok(out)
    }

    pub fn download_neighs_local(&self) -> ocl::Result<Vec<i32>> {
        let mut out = vec![0i32; self.natoms * 4];
        self.cl_neighs_local.read(&mut out[..]).enq()?;
        Ok(out)
    }

    pub fn download_excl_local(&self) -> ocl::Result<(Vec<i32>, Vec<i32>)> {
        let mut e1 = vec![0i32; self.natoms * 4];
        let mut e2 = vec![0i32; self.natoms * 4];
        self.cl_excl1_local.read(&mut e1[..]).enq()?;
        self.cl_excl2_local.read(&mut e2[..]).enq()?;
        Ok((e1, e2))
    }

    pub fn download_dpos_coll(&self) -> ocl::Result<Vec<[f32; 4]>> {
        let mut out = vec![0.0f32; self.natoms * 4];
        self.cl_dpos_coll.read(&mut out[..]).enq()?;
        Ok((0..self.natoms).map(|i| [out[i*4], out[i*4+1], out[i*4+2], out[i*4+3]]).collect())
    }

    pub fn download_dpos_node(&self) -> ocl::Result<Vec<[f32; 4]>> {
        let buf = self.cl_dpos_node.as_ref().expect("download_dpos_node: node buffers not allocated");
        let mut out = vec![0.0f32; self.nnode_tot * 4];
        buf.read(&mut out[..]).enq()?;
        Ok((0..self.nnode_tot).map(|i| [out[i*4], out[i*4+1], out[i*4+2], out[i*4+3]]).collect())
    }

    pub fn download_drot_node(&self) -> ocl::Result<Vec<[f32; 4]>> {
        let buf = self.cl_drot_node.as_ref().expect("download_drot_node: node buffers not allocated");
        let mut out = vec![0.0f32; self.nnode_tot * 4];
        buf.read(&mut out[..]).enq()?;
        Ok((0..self.nnode_tot).map(|i| [out[i*4], out[i*4+1], out[i*4+2], out[i*4+3]]).collect())
    }

    pub fn download_dpos_neigh(&self) -> ocl::Result<Vec<[f32; 4]>> {
        let buf = self.cl_dpos_neigh.as_ref().expect("download_dpos_neigh: node buffers not allocated");
        let n = self.nnode_tot * 4;
        let mut out = vec![0.0f32; n * 4];
        buf.read(&mut out[..]).enq()?;
        Ok((0..n).map(|i| [out[i*4], out[i*4+1], out[i*4+2], out[i*4+3]]).collect())
    }

    // ------------------------------------------------------------------
    //  Internal: kernel dispatch helpers
    // ------------------------------------------------------------------

    fn margin_sq(&self, bbox_margin: f32) -> ocl::Result<f32> {
        let mut rad = vec![0.0f32; self.natoms];
        self.cl_radius.read(&mut rad[..]).enq()?;
        let rmax = rad.iter().copied().fold(0.0f32, f32::max);
        Ok((2.0 * rmax + bbox_margin).powi(2))
    }

    fn zero_corrections(&self) -> ocl::Result<()> {
        let zeros_n = vec![0.0f32; self.natoms * 4];
        let zeros_nn = vec![0.0f32; self.nnode_tot * 4];
        let zeros_nn4 = vec![0.0f32; self.nnode_tot * 4 * 4];
        self.cl_dpos_coll.write(&zeros_n[..]).enq()?;
        if let Some(ref b) = self.cl_dpos_node { b.write(&zeros_nn[..]).enq()?; }
        if let Some(ref b) = self.cl_drot_node { b.write(&zeros_nn[..]).enq()?; }
        if let Some(ref b) = self.cl_dpos_neigh { b.write(&zeros_nn4[..]).enq()?; }
        Ok(())
    }

    // ------------------------------------------------------------------
    //  Broad-phase kernels
    // ------------------------------------------------------------------

    fn run_bboxes_and_topology(&self, bbox_margin: f32) -> ocl::Result<()> {
        let gs = self.natoms;
        let ls = self.group_size;
        let margin_sq = self.margin_sq(bbox_margin)?;
        let local_floats = self.group_size * 4; // float4 elements = 4 floats each
        let k1 = Kernel::builder()
            .program(&self.program).name("update_bboxes_rigid").queue(self.queue.clone())
            .global_work_size(gs).local_work_size(ls)
            .arg(&self.cl_pos).arg(&self.cl_radius)
            .arg(&self.cl_bboxes_min).arg(&self.cl_bboxes_max)
            .arg_local::<f32>(local_floats).arg_local::<f32>(local_floats)
            .arg(self.natoms as i32)
            .build()?;
        unsafe { k1.enq()?; }
        let k2 = Kernel::builder()
            .program(&self.program).name("build_local_topology_rigid").queue(self.queue.clone())
            .global_work_size(gs).local_work_size(ls)
            .arg(&self.cl_pos).arg(&self.cl_bboxes_min).arg(&self.cl_bboxes_max)
            .arg(&self.cl_neighs).arg(&self.cl_excl1).arg(&self.cl_excl2)
            .arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
            .arg(&self.cl_neighs_local).arg(&self.cl_excl1_local).arg(&self.cl_excl2_local)
            .arg(self.natoms as i32).arg(self.num_groups as i32)
            .arg(margin_sq).arg(bbox_margin)
            .build()?;
        unsafe { k2.enq()?; }
        Ok(())
    }

    fn run_collision(&self, k_coll: f32) -> ocl::Result<()> {
        let k = Kernel::builder()
            .program(&self.program).name("compute_collision_cluster_rigid").queue(self.queue.clone())
            .global_work_size(self.natoms).local_work_size(self.group_size)
            .arg(&self.cl_pos).arg(&self.cl_radius)
            .arg(&self.cl_excl1_local).arg(&self.cl_excl2_local)
            .arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
            .arg(&self.cl_dpos_coll).arg(self.natoms as i32).arg(k_coll)
            .build()?;
        unsafe { k.enq()?; }
        Ok(())
    }

    // ------------------------------------------------------------------
    //  Port kernels (5 variants)
    // ------------------------------------------------------------------

    fn run_ports(&mut self, port_kernel: PortKernel, cfg: &StepConfig) -> ocl::Result<()> {
        let gs = self.natoms;
        let ls = self.group_size;
        let nnode = self.nnode_per_group;
        let pl = self.cl_port_local.as_ref().unwrap();
        let kf = self.cl_kflat.as_ref().unwrap();
        let dpn = self.cl_dpos_node.as_ref().unwrap();
        let drn = self.cl_drot_node.as_ref().unwrap();
        let dpneigh = self.cl_dpos_neigh.as_ref().unwrap();

        match port_kernel {
            PortKernel::Current => {
                let k = Kernel::builder()
                    .program(&self.program).name("compute_ports_cluster_rigid").queue(self.queue.clone())
                    .global_work_size(gs).local_work_size(ls)
                    .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_radius)
                    .arg(&self.cl_neighs_local).arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
                    .arg(pl).arg(kf).arg(dpn).arg(drn).arg(dpneigh)
                    .arg(self.natoms as i32).arg(nnode).arg(cfg.dt).arg(0i32).arg(cfg.rot_mass_scale)
                    .build()?;
                unsafe { k.enq()?; }
            }
            PortKernel::Orig => {
                let k = Kernel::builder()
                    .program(&self.program).name("compute_ports_cluster_rigid_orig").queue(self.queue.clone())
                    .global_work_size(gs).local_work_size(ls)
                    .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_radius)
                    .arg(&self.cl_neighs_local).arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
                    .arg(pl).arg(kf).arg(dpn).arg(drn).arg(dpneigh)
                    .arg(self.natoms as i32).arg(nnode).arg(cfg.dt).arg(0i32)
                    .arg(None::<&Buffer<f32>>).arg(0i32)  // quat_opt=null, skip_rotation=0
                    .build()?;
                unsafe { k.enq()?; }
            }
            PortKernel::Substep => {
                let k = Kernel::builder()
                    .program(&self.program).name("compute_ports_cluster_rigid_substep_optimized").queue(self.queue.clone())
                    .global_work_size(gs).local_work_size(ls)
                    .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_radius)
                    .arg(&self.cl_neighs_local).arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
                    .arg(pl).arg(kf).arg(dpn).arg(drn).arg(dpneigh)
                    .arg(self.natoms as i32).arg(nnode).arg(cfg.dt).arg(0i32)
                    .arg(cfg.n_rot_substeps).arg(cfg.rot_eps).arg(cfg.theta_max)
                    .build()?;
                unsafe { k.enq()?; }
            }
            PortKernel::Shapematch => {
                let k = Kernel::builder()
                    .program(&self.program).name("compute_ports_cluster_rigid_shapematch").queue(self.queue.clone())
                    .global_work_size(gs).local_work_size(ls)
                    .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_radius)
                    .arg(&self.cl_neighs_local).arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
                    .arg(pl).arg(kf).arg(dpn).arg(drn).arg(dpneigh)
                    .arg(self.natoms as i32).arg(nnode).arg(cfg.dt).arg(0i32)
                    .build()?;
                unsafe { k.enq()?; }
            }
            PortKernel::Eigen => {
                if !self.tips_valid {
                    let tips = self.cl_tips.as_ref().unwrap();
                    let kt = Kernel::builder()
                        .program(&self.program).name("compute_tips").queue(self.queue.clone())
                        .global_work_size(self.natoms)
                        .arg(self.natoms as i32).arg(nnode)
                        .arg(&self.cl_pos).arg(&self.cl_quat).arg(pl).arg(tips)
                        .build()?;
                    unsafe { kt.enq()?; }
                    self.tips_valid = true;
                }
                let tips = self.cl_tips.as_ref().unwrap();
                let k1 = Kernel::builder()
                    .program(&self.program).name("compute_optimal_rotation_eigen").queue(self.queue.clone())
                    .global_work_size(gs).local_work_size(ls)
                    .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_radius)
                    .arg(&self.cl_neighs_local).arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
                    .arg(pl).arg(kf).arg(drn).arg(&self.cl_dquat_mom)
                    .arg(self.natoms as i32).arg(nnode).arg(cfg.dt)
                    .build()?;
                unsafe { k1.enq()?; }
                let k2 = Kernel::builder()
                    .program(&self.program).name("compute_ports_cluster_rigid_eigen_tips").queue(self.queue.clone())
                    .global_work_size(gs).local_work_size(ls)
                    .arg(&self.cl_pos).arg(&self.cl_neighs_local)
                    .arg(&self.cl_ghost_indices).arg(&self.cl_ghost_counts)
                    .arg(kf).arg(tips).arg(dpn).arg(dpneigh)
                    .arg(self.natoms as i32).arg(nnode).arg(cfg.dt)
                    .build()?;
                unsafe { k2.enq()?; }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    //  Corrections + dynamics
    // ------------------------------------------------------------------

    fn run_corrections(&self, cfg: &StepConfig, massless_rot: i32) -> ocl::Result<()> {
        // Kernel always expects port_local + tips args (checks null internally for massless_rot).
        // We always pass the real buffers since they're allocated by ensure_node_buffers.
        let k = Kernel::builder()
            .program(&self.program).name("apply_corrections_rigid_ports").queue(self.queue.clone())
            .global_work_size(self.natoms)
            .arg(self.natoms as i32).arg(self.nnode_per_group)
            .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_fixmask)
            .arg(self.cl_bk_slots.as_ref().unwrap())
            .arg(self.cl_dpos_node.as_ref().unwrap())
            .arg(self.cl_drot_node.as_ref().unwrap())
            .arg(self.cl_dpos_neigh.as_ref().unwrap())
            .arg(&self.cl_dpos_coll).arg(&self.cl_dpos_mom).arg(&self.cl_dquat_mom)
            .arg(cfg.relaxation).arg(cfg.momentum_beta).arg(massless_rot)
            .arg(self.cl_port_local.as_ref().unwrap())
            .arg(self.cl_tips.as_ref().unwrap())
            .build()?;
        unsafe { k.enq()?; }
        Ok(())
    }

    fn run_predict(&self, dt: f32) -> ocl::Result<()> {
        let k = Kernel::builder()
            .program(&self.program).name("predict_dynamics").queue(self.queue.clone())
            .global_work_size(self.natoms)
            .arg(self.natoms as i32).arg(self.nnode_per_group)
            .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_fixmask)
            .arg(&self.cl_vel).arg(&self.cl_omega)
            .arg(&self.cl_pos_prev).arg(&self.cl_quat_prev).arg(dt)
            .build()?;
        unsafe { k.enq()?; }
        Ok(())
    }

    fn run_update_velocities(&self, dt: f32, damp: f32) -> ocl::Result<()> {
        let k = Kernel::builder()
            .program(&self.program).name("update_velocities_dynamics").queue(self.queue.clone())
            .global_work_size(self.natoms)
            .arg(self.natoms as i32).arg(self.nnode_per_group)
            .arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_fixmask)
            .arg(&self.cl_vel).arg(&self.cl_omega)
            .arg(&self.cl_pos_prev).arg(&self.cl_quat_prev).arg(dt).arg(damp)
            .build()?;
        unsafe { k.enq()?; }
        Ok(())
    }

    // ------------------------------------------------------------------
    //  Public step functions
    // ------------------------------------------------------------------

    /// One cluster (relaxation) step. Runs: bboxes -> topology -> collision -> ports -> corrections.
    /// Port of RRsp3.py::step_cluster.
    pub fn step_cluster(&mut self, port_kernel: PortKernel, cfg: &StepConfig) -> ocl::Result<()> {
        self.run_bboxes_and_topology(cfg.bbox_margin)?;
        self.zero_corrections()?;
        self.run_collision(cfg.k_coll)?;
        self.run_ports(port_kernel, cfg)?;
        let massless = port_kernel.is_massless() as i32;
        self.run_corrections(cfg, massless)?;
        Ok(())
    }

    /// One dynamics step (leapfrog/PBD-style). Runs: predict -> bboxes -> topology ->
    /// collision -> ports -> corrections -> update_velocities.
    /// Port of RRsp3.py::step_dynamics.
    pub fn step_dynamics(&mut self, port_kernel: PortKernel, cfg: &StepConfig) -> ocl::Result<()> {
        self.reset_momentum()?;
        self.tips_valid = false;
        self.run_predict(cfg.dt)?;
        self.run_bboxes_and_topology(cfg.bbox_margin)?;
        self.zero_corrections()?;
        self.run_collision(cfg.k_coll)?;
        self.run_ports(port_kernel, cfg)?;
        let massless = port_kernel.is_massless() as i32;
        let mut cfg2 = *cfg;
        cfg2.momentum_beta = 0.0;
        self.run_corrections(&cfg2, massless)?;
        self.run_update_velocities(cfg.dt, cfg.damp)?;
        Ok(())
    }
}

// ==================================================================
// RRsp3Multi — multi-replica harness for independent replicas
//
// 2D NDRange: global = (GROUP_SIZE, nSys), local = (GROUP_SIZE, 1)
// One workgroup per replica. No ghosts (all neighbors intra-workgroup).
// Shared topology (neighs, excl, port_local, bk_slots, radius) uploaded once.
// Per-replica state (pos, quat, dpos_*) sized nSys * natoms.
//
// Kernels in RRsp3.cl: compute_collision_multi, compute_ports_current_multi,
//   apply_corrections_multi, zero_corrections_multi
// ==================================================================

/// Multi-replica OpenCL harness for independent replicas of the same molecule.
/// Uses 2D NDRange (atom, replica). No ghosts — all neighbors are intra-workgroup.
pub struct RRsp3Multi {
    pub natoms_per_sys: usize,   // atoms per replica (padded to group_size)
    pub nnode_per_group: i32,    // nodes per replica
    pub nsys: usize,             // number of independent replicas
    pub group_size: usize,

    queue: Queue,
    program: Program,
    device_name: String,

    // Shared topology buffers (uploaded once, same for all replicas)
    cl_radius: Buffer<f32>,      // [natoms_per_sys]
    cl_neighs: Buffer<i32>,      // [natoms_per_sys * 4] int4
    cl_excl1: Buffer<i32>,       // [natoms_per_sys * 4] int4
    cl_excl2: Buffer<i32>,       // [natoms_per_sys * 4] int4
    cl_fixmask: Buffer<i32>,     // [natoms_per_sys]
    cl_port_local: Buffer<f32>,  // [nnode * 4 * 4] float4
    cl_kflat: Buffer<f32>,       // [nnode * 4]
    cl_bk_slots: Buffer<i32>,    // [natoms_per_sys * 4] int4

    // Per-replica state buffers (sized nsys * natoms_per_sys)
    cl_pos: Buffer<f32>,         // [nsys * natoms * 4] float4
    cl_quat: Buffer<f32>,        // [nsys * natoms * 4] float4
    cl_dpos_coll: Buffer<f32>,   // [nsys * natoms * 4]
    cl_dpos_mom: Buffer<f32>,    // [nsys * natoms * 4]
    cl_dquat_mom: Buffer<f32>,   // [nsys * natoms * 4]

    // Per-replica node buffers (sized nsys * nnode)
    cl_dpos_node: Buffer<f32>,   // [nsys * nnode * 4]
    cl_drot_node: Buffer<f32>,   // [nsys * nnode * 4]
    cl_dpos_neigh: Buffer<f32>,  // [nsys * nnode * 4 * 4]
    cl_tips: Buffer<f32>,        // [nsys * nnode * 4 * 4] (for massless variants)

    // Persistent kernels, built after node buffers are allocated.
    k_collision: Option<Kernel>,
    k_ports_current: Option<Kernel>,
    k_corrections: Option<Kernel>,
    collisions_enabled: bool,
}

impl RRsp3Multi {
    /// Create a multi-replica harness. `natoms_per_sys` must be <= group_size
    /// (one workgroup per replica). `nsys` = number of independent replicas.
    pub fn new(natoms_per_sys: usize, group_size: usize, nsys: usize, prefer_gpu: bool) -> ocl::Result<Self> {
        assert!(natoms_per_sys <= group_size, "RRsp3Multi::new: natoms_per_sys={natoms_per_sys} must be <= group_size={group_size} (one workgroup per replica)");
        assert!(group_size.is_power_of_two(), "RRsp3Multi::new: group_size={group_size} must be power of two");
        assert!(nsys > 0, "RRsp3Multi::new: nsys must be > 0");

        // Select platform + device (same logic as RRsp3::new)
        let (platform, device) = if prefer_gpu {
            let mut found = None;
            for p in Platform::list() {
                if let Ok(gpus) = Device::list(&p, Some(flags::DEVICE_TYPE_GPU)) {
                    for d in gpus {
                        if let Ok(DeviceInfoResult::Vendor(v)) = d.info(DeviceInfo::Vendor) {
                            if v.contains("NVIDIA") { found = Some((p, d)); break; }
                        }
                    }
                }
                if found.is_some() { break; }
            }
            if let Some(pd) = found { pd }
            else {
                let dp = Platform::default();
                let d = Device::list(&dp, Some(flags::DEVICE_TYPE_GPU))
                    .ok().and_then(|g| g.into_iter().next())
                    .unwrap_or_else(|| Device::first(&dp).expect("RRsp3Multi::new: no OpenCL device"));
                (dp, d)
            }
        } else {
            let dp = Platform::default();
            (dp, Device::first(&dp)?)
        };
        let device_name = device.name()?;

        let context = Context::builder().platform(platform).devices(device.clone()).build()?;
        let queue = Queue::new(&context, device.clone(), None)?;

        let program = ProgramBuilder::new()
            .devices(device.clone()).src(RRSP3_SRC)
            .cmplr_def("GROUP_SIZE", group_size as i32)
            .cmplr_def("MAX_GHOSTS", 1i32)  // unused for multi, but required by kernel
            .build(&context)?;

        let mk_fbuf = |len: usize, fl: MemFlags| -> ocl::Result<Buffer<f32>> {
            Buffer::<f32>::builder().queue(queue.clone()).flags(fl).len(len).build()
        };
        let mk_ibuf = |len: usize, fl: MemFlags| -> ocl::Result<Buffer<i32>> {
            Buffer::<i32>::builder().queue(queue.clone()).flags(fl).len(len).build()
        };

        let n = natoms_per_sys;
        let ns = nsys;
        let n4 = n * 4;
        let ns_n4 = ns * n4;

        // Shared topology
        let cl_radius = mk_fbuf(n, flags::MEM_READ_ONLY)?;
        let cl_neighs = mk_ibuf(n4, flags::MEM_READ_ONLY)?;
        let cl_excl1 = mk_ibuf(n4, flags::MEM_READ_ONLY)?;
        let cl_excl2 = mk_ibuf(n4, flags::MEM_READ_ONLY)?;
        let cl_fixmask = mk_ibuf(n, flags::MEM_READ_ONLY)?;
        // Node-only shared buffers: allocated lazily when nnode_per_group is set
        // For now, allocate with size 0; will be reallocated in upload_cluster_ports_multi
        let cl_port_local = mk_fbuf(1, flags::MEM_READ_ONLY)?;
        let cl_kflat = mk_fbuf(1, flags::MEM_READ_ONLY)?;
        let cl_bk_slots = mk_ibuf(n4, flags::MEM_READ_ONLY)?;

        // Per-replica state
        let cl_pos = mk_fbuf(ns_n4, flags::MEM_READ_WRITE)?;
        let cl_quat = mk_fbuf(ns_n4, flags::MEM_READ_WRITE)?;
        let cl_dpos_coll = mk_fbuf(ns_n4, flags::MEM_READ_WRITE)?;
        let cl_dpos_mom = mk_fbuf(ns_n4, flags::MEM_READ_WRITE)?;
        let cl_dquat_mom = mk_fbuf(ns_n4, flags::MEM_READ_WRITE)?;

        // Per-replica node buffers: allocated with size 0, reallocated in upload_cluster_ports_multi
        let cl_dpos_node = mk_fbuf(1, flags::MEM_READ_WRITE)?;
        let cl_drot_node = mk_fbuf(1, flags::MEM_READ_WRITE)?;
        let cl_dpos_neigh = mk_fbuf(1, flags::MEM_READ_WRITE)?;
        let cl_tips = mk_fbuf(1, flags::MEM_READ_WRITE)?;

        // Zero persistent state that may be read before its producer is enabled.
        let zeros = vec![0.0f32; ns_n4];
        cl_dpos_coll.write(&zeros[..]).enq()?;
        cl_dpos_mom.write(&zeros[..]).enq()?;
        cl_dquat_mom.write(&zeros[..]).enq()?;
        let zeros_i = vec![0i32; n];
        cl_fixmask.write(&zeros_i[..]).enq()?;

        Ok(Self {
            natoms_per_sys: n, nnode_per_group: 0, nsys: ns, group_size,
            queue, program, device_name,
            cl_radius, cl_neighs, cl_excl1, cl_excl2, cl_fixmask,
            cl_port_local, cl_kflat, cl_bk_slots,
            cl_pos, cl_quat, cl_dpos_coll, cl_dpos_mom, cl_dquat_mom,
            cl_dpos_node, cl_drot_node, cl_dpos_neigh, cl_tips,
            k_collision: None, k_ports_current: None, k_corrections: None,
            collisions_enabled: false,
        })
    }

    pub fn device_name(&self) -> &str { &self.device_name }
    pub fn nsys(&self) -> usize { self.nsys }

    fn build_kernels(&mut self) -> ocl::Result<()> {
        let n = self.natoms_per_sys as i32;
        let nn = self.nnode_per_group;
        let gs = self.group_size;
        let ns = self.nsys;
        self.k_collision = Some(Kernel::builder().program(&self.program).name("compute_collision_multi").queue(self.queue.clone()).global_work_size((gs, ns)).local_work_size((gs, 1)).arg(&self.cl_pos).arg(&self.cl_radius).arg(&self.cl_excl1).arg(&self.cl_excl2).arg(&self.cl_dpos_coll).arg(n).arg(0.0f32).build()?);
        self.k_ports_current = Some(Kernel::builder().program(&self.program).name("compute_ports_current_multi").queue(self.queue.clone()).global_work_size((gs, ns)).local_work_size((gs, 1)).arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_radius).arg(&self.cl_neighs).arg(&self.cl_port_local).arg(&self.cl_kflat).arg(&self.cl_dpos_node).arg(&self.cl_drot_node).arg(&self.cl_dpos_neigh).arg(n).arg(nn).arg(0.0f32).arg(0.0f32).build()?);
        self.k_corrections = Some(Kernel::builder().program(&self.program).name("apply_corrections_multi").queue(self.queue.clone()).global_work_size((gs, ns)).local_work_size((gs, 1)).arg(n).arg(nn).arg(&self.cl_pos).arg(&self.cl_quat).arg(&self.cl_fixmask).arg(&self.cl_bk_slots).arg(&self.cl_dpos_node).arg(&self.cl_drot_node).arg(&self.cl_dpos_neigh).arg(&self.cl_dpos_coll).arg(&self.cl_dpos_mom).arg(&self.cl_dquat_mom).arg(0.0f32).arg(0.0f32).arg(0i32).arg(&self.cl_port_local).arg(&self.cl_tips).build()?);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Upload shared topology (same for all replicas)
    // ------------------------------------------------------------------

    pub fn upload_radius(&mut self, radius: &[f32]) -> ocl::Result<()> {
        assert!(radius.len() == self.natoms_per_sys, "upload_radius: len={} != natoms_per_sys={}", radius.len(), self.natoms_per_sys);
        for (i, &r) in radius.iter().enumerate() { assert!(r.is_finite() && r >= 0.0, "upload_radius: radius[{i}] must be finite and non-negative, got {r}"); }
        self.collisions_enabled = radius.iter().any(|&r| r > 0.0);
        self.cl_radius.write(radius).enq()?;
        if !self.collisions_enabled {
            let zeros = vec![0.0f32; self.nsys * self.natoms_per_sys * 4];
            self.cl_dpos_coll.write(&zeros[..]).enq()?;
        }
        Ok(())
    }

    pub fn upload_neighs_and_exclusions(&mut self, neighs: &[i32], excl1: &[i32], excl2: &[i32]) -> ocl::Result<()> {
        assert!(neighs.len() == self.natoms_per_sys * 4, "upload_neighs: len={}", neighs.len());
        assert!(excl1.len() == self.natoms_per_sys * 4, "upload_excl1: len={}", excl1.len());
        assert!(excl2.len() == self.natoms_per_sys * 4, "upload_excl2: len={}", excl2.len());
        self.cl_neighs.write(&neighs[..]).enq()?;
        self.cl_excl1.write(&excl1[..]).enq()?;
        self.cl_excl2.write(&excl2[..]).enq()
    }

    pub fn upload_fixmask(&mut self, fixmask: &[i32]) -> ocl::Result<()> {
        assert!(fixmask.len() == self.natoms_per_sys, "upload_fixmask: len={}", fixmask.len());
        self.cl_fixmask.write(&fixmask[..]).enq()
    }

    /// Upload cluster port geometry + stiffness (shared). `port_local_atoms` is per-atom
    /// [natoms * 4 * 4] (float4 per port), `k_atoms` is per-atom [natoms * 4].
    /// We extract only the node rows (first nnode_per_group atoms) and pack as [nnode * 4 * 4].
    pub fn upload_cluster_ports_multi(&mut self, port_local_atoms: &[f32], k_atoms: &[f32], nnode_per_group: i32) -> ocl::Result<()> {
        assert!(nnode_per_group >= 0 && nnode_per_group as usize <= self.group_size);
        assert!(port_local_atoms.len() == self.natoms_per_sys * 4 * 4, "upload_cluster_ports_multi: port_local len={}", port_local_atoms.len());
        assert!(k_atoms.len() == self.natoms_per_sys * 4, "upload_cluster_ports_multi: k_atoms len={}", k_atoms.len());
        self.nnode_per_group = nnode_per_group;
        let nnode = nnode_per_group as usize;
        let ns = self.nsys;

        // Reallocate node buffers with correct size
        let q = self.queue.clone();
        let mk_fbuf = |len: usize, fl: MemFlags| -> ocl::Result<Buffer<f32>> {
            Buffer::<f32>::builder().queue(q.clone()).flags(fl).len(len).build()
        };
        self.cl_port_local = mk_fbuf(nnode * 4 * 4, flags::MEM_READ_ONLY)?;
        self.cl_kflat = mk_fbuf(nnode * 4, flags::MEM_READ_ONLY)?;
        self.cl_dpos_node = mk_fbuf(ns * nnode * 4, flags::MEM_READ_WRITE)?;
        self.cl_drot_node = mk_fbuf(ns * nnode * 4, flags::MEM_READ_WRITE)?;
        self.cl_dpos_neigh = mk_fbuf(ns * nnode * 4 * 4, flags::MEM_READ_WRITE)?;
        self.cl_tips = mk_fbuf(ns * nnode * 4 * 4, flags::MEM_READ_WRITE)?;

        // Pack port_local: extract node rows (atoms 0..nnode-1) → [nnode * 4 * 4]
        let mut pl = vec![0.0f32; nnode * 4 * 4];
        let mut kk = vec![0.0f32; nnode * 4];
        for il in 0..nnode {
            for k in 0..4 {
                let src = (il * 4 + k) * 4;
                let dst = (il * 4 + k) * 4;
                pl[dst..dst + 4].copy_from_slice(&port_local_atoms[src..src + 4]);
                kk[il * 4 + k] = k_atoms[il * 4 + k];
            }
        }
        self.cl_port_local.write(&pl[..]).enq()?;
        self.cl_kflat.write(&kk[..]).enq()?;
        self.build_kernels()
    }

    /// Upload bk_slots (shared). [natoms * 4] int4 — local slot indices into dpos_neigh.
    pub fn upload_bk_slots_multi(&mut self, bk_slots: &[i32]) -> ocl::Result<()> {
        assert!(bk_slots.len() == self.natoms_per_sys * 4, "upload_bk_slots_multi: len={}", bk_slots.len());
        self.cl_bk_slots.write(&bk_slots[..]).enq()
    }

    // ------------------------------------------------------------------
    // Upload per-replica state
    // ------------------------------------------------------------------

    /// Upload positions + inverse masses for ALL replicas.
    /// `pos3_flat` = [nsys * natoms * 3], `inv_mass` = [natoms] (shared, same molecule).
    /// `quat_flat` = optional [nsys * natoms * 4], defaults to identity.
    pub fn upload_state_multi(&mut self, pos3_flat: &[f32], inv_mass: &[f32], quat_flat: Option<&[f32]>) -> ocl::Result<()> {
        let n = self.natoms_per_sys;
        let ns = self.nsys;
        assert!(pos3_flat.len() == ns * n * 3, "upload_state_multi: pos3_flat len={} != {}*{}*3", pos3_flat.len(), ns, n);
        assert!(inv_mass.len() == n, "upload_state_multi: inv_mass len={}", inv_mass.len());

        let mut pos4 = vec![0.0f32; ns * n * 4];
        for is in 0..ns {
            for i in 0..n {
                let src = (is * n + i) * 3;
                let dst = (is * n + i) * 4;
                pos4[dst + 0] = pos3_flat[src + 0];
                pos4[dst + 1] = pos3_flat[src + 1];
                pos4[dst + 2] = pos3_flat[src + 2];
                pos4[dst + 3] = inv_mass[i];
                if inv_mass[i] <= 1e-12 { pos4[dst + 0] = f32::NAN; pos4[dst + 1] = f32::NAN; pos4[dst + 2] = f32::NAN; }
            }
        }
        self.cl_pos.write(&pos4[..]).enq()?;

        let mut quat4 = vec![0.0f32; ns * n * 4];
        if let Some(q) = quat_flat {
            assert!(q.len() == ns * n * 4, "upload_state_multi: quat len={}", q.len());
            quat4.copy_from_slice(q);
        } else {
            for i in 0..ns * n { quat4[i * 4 + 3] = 1.0; }
        }
        self.cl_quat.write(&quat4[..]).enq()?;

        // Zero momentum
        let zeros = vec![0.0f32; ns * n * 4];
        self.cl_dpos_mom.write(&zeros[..]).enq()?;
        self.cl_dquat_mom.write(&zeros[..]).enq()
    }

    // ------------------------------------------------------------------
    // Download
    // ------------------------------------------------------------------

    /// Download positions for ALL replicas. Returns [nsys * natoms * 3] flat f32.
    pub fn download_pos_multi(&self) -> ocl::Result<Vec<f32>> {
        let n = self.natoms_per_sys;
        let ns = self.nsys;
        let mut out4 = vec![0.0f32; ns * n * 4];
        self.cl_pos.read(&mut out4[..]).enq()?;
        let mut out3 = vec![0.0f32; ns * n * 3];
        for i in 0..ns * n {
            out3[i * 3 + 0] = out4[i * 4 + 0];
            out3[i * 3 + 1] = out4[i * 4 + 1];
            out3[i * 3 + 2] = out4[i * 4 + 2];
        }
        Ok(out3)
    }

    /// Download positions for one replica. Returns [natoms * 3] flat f32.
    pub fn download_pos_replica(&self, isys: usize) -> ocl::Result<Vec<f32>> {
        let n = self.natoms_per_sys;
        let ns = self.nsys;
        assert!(isys < ns, "download_pos_replica: isys={isys} >= nsys={ns}");
        let mut all = vec![0.0f32; ns * n * 4];
        self.cl_pos.read(&mut all[..]).enq()?;
        let mut out = vec![0.0f32; n * 3];
        for i in 0..n {
            let src = (isys * n + i) * 4;
            out[i * 3 + 0] = all[src + 0];
            out[i * 3 + 1] = all[src + 1];
            out[i * 3 + 2] = all[src + 2];
        }
        Ok(out)
    }

    /// Download quaternions for one replica. Returns [natoms * 4] flat f32.
    pub fn download_quat_replica(&self, isys: usize) -> ocl::Result<Vec<f32>> {
        let n = self.natoms_per_sys;
        let ns = self.nsys;
        assert!(isys < ns, "download_quat_replica: isys={isys} >= nsys={ns}");
        let mut all = vec![0.0f32; ns * n * 4];
        self.cl_quat.read(&mut all[..]).enq()?;
        let mut out = vec![0.0f32; n * 4];
        for i in 0..n { out[i * 4..i * 4 + 4].copy_from_slice(&all[(isys * n + i) * 4..(isys * n + i) * 4 + 4]); }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Step (multi-replica)
    // ------------------------------------------------------------------

    /// One cluster (relaxation) step for ALL replicas.
    /// Runs: [collision when any radius > 0] -> ports -> corrections. Current-mode producer kernels
    /// overwrite every consumed correction slot; radius-zero collision output is initialized once.
    /// No bboxes/topology (independent replicas, no ghosts).
    /// Currently supports PortKernel::Current only.
    pub fn step_cluster_multi(&mut self, port_kernel: PortKernel, cfg: &StepConfig) -> ocl::Result<()> {
        assert!(port_kernel == PortKernel::Current, "step_cluster_multi: PortKernel::{port_kernel:?} not yet implemented for multi-replica (Current only)");
        let kc = self.k_collision.as_ref().expect("step_cluster_multi: call upload_cluster_ports_multi first; persistent kernels are not built");
        let kp = self.k_ports_current.as_ref().expect("step_cluster_multi: Current kernel is not built");
        let kr = self.k_corrections.as_ref().expect("step_cluster_multi: corrections kernel is not built");

        if self.collisions_enabled {
            kc.set_arg(6, cfg.k_coll)?;
            unsafe { kc.enq()?; }
        }

        kp.set_arg(11, cfg.dt)?;
        kp.set_arg(12, cfg.rot_mass_scale)?;
        unsafe { kp.enq()?; }

        kr.set_arg(12, cfg.relaxation)?;
        kr.set_arg(13, cfg.momentum_beta)?;
        kr.set_arg(14, 0i32)?;
        unsafe { kr.enq()?; }
        Ok(())
    }

    /// Download dpos_coll for one replica (max|correction| = convergence metric).
    pub fn download_dpos_coll_replica(&self, isys: usize) -> ocl::Result<Vec<f32>> {
        let n = self.natoms_per_sys;
        let ns = self.nsys;
        assert!(isys < ns, "download_dpos_coll_replica: isys={isys} >= nsys={ns}");
        let mut all = vec![0.0f32; ns * n * 4];
        self.cl_dpos_coll.read(&mut all[..]).enq()?;
        let mut out = vec![0.0f32; n * 3];
        for i in 0..n {
            let src = (isys * n + i) * 4;
            out[i * 3 + 0] = all[src + 0];
            out[i * 3 + 1] = all[src + 1];
            out[i * 3 + 2] = all[src + 2];
        }
        Ok(out)
    }

    /// Download dpos_node for one replica.
    pub fn download_dpos_node_replica(&self, isys: usize) -> ocl::Result<Vec<f32>> {
        let nnode = self.nnode_per_group as usize;
        let ns = self.nsys;
        assert!(isys < ns, "download_dpos_node_replica: isys={isys} >= nsys={ns}");
        let mut all = vec![0.0f32; ns * nnode * 4];
        self.cl_dpos_node.read(&mut all[..]).enq()?;
        let mut out = vec![0.0f32; nnode * 3];
        for i in 0..nnode {
            let src = (isys * nnode + i) * 4;
            out[i * 3 + 0] = all[src + 0];
            out[i * 3 + 1] = all[src + 1];
            out[i * 3 + 2] = all[src + 2];
        }
        Ok(out)
    }

    /// Finish the queue (wait for all pending kernels).
    pub fn finish(&self) -> ocl::Result<()> {
        self.queue.finish()
    }
}
