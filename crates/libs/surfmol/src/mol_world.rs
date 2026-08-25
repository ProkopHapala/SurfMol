
use moltopo::molecular::DynamicAtoms;
use moltopo::topology::Topology;
use molff::uff::Uff;
use molff::nonbonded::NonBondedFF;
use surfff::{SurfaceFolded, SurfaceScratch};
use molff::rigid_sp3::RigidSp3FF;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BondedFFMode { Uff, RigidSp3 }

/// MolWorld orchestrates multiple forcefield engines for molecular dynamics.
/// Does NOT own apos/fapos/vapos directly — those live in DynamicAtoms.
/// Uff owns bonded topology; NonBondedFF owns REQH/PLQH; Surface owns substrate geometry.
pub struct MolWorld {
    pub dyn_atoms: DynamicAtoms,
    pub uff: Uff,
    pub rigid_sp3: RigidSp3FF,
    pub bonded_mode: BondedFFMode,
    pub nonbonded: Option<NonBondedFF>,
    pub surface: Option<SurfaceFolded>,
}

impl MolWorld {
    pub fn from_topology(top: &Topology) -> Self {
        let uff = Uff::from_topology(top);
        let natoms = uff.natoms;
        let mut dyn_atoms = DynamicAtoms::new(natoms);
        dyn_atoms.atoms.apos.as_mut_slice().copy_from_slice(&top.apos);
        dyn_atoms.atoms.make_neigh_bs(uff.bon_atoms.as_slice());
        Self {
            dyn_atoms,
            uff,
            rigid_sp3: RigidSp3FF::from_uff(natoms as usize),
            bonded_mode: BondedFFMode::RigidSp3,
            nonbonded: None,
            surface: None,
        }
    }

    pub fn from_uff(uff: Uff) -> Self {
        let natoms = uff.natoms;
        let mut dyn_atoms = DynamicAtoms::new(natoms);
        dyn_atoms.atoms.make_neigh_bs(uff.bon_atoms.as_slice());
        Self {
            dyn_atoms,
            uff,
            rigid_sp3: RigidSp3FF::from_uff(natoms as usize),
            bonded_mode: BondedFFMode::RigidSp3,
            nonbonded: None,
            surface: None,
        }
    }

    pub fn natoms(&self) -> usize { self.dyn_atoms.natoms() }

    /// Evaluate all forces, accumulating into dyn_atoms.fapos.
    /// Returns: (eb, ea, ed, ei, enb, es) = (bond, angle, dihedral, inversion, nonbonded, surface)
    pub fn eval_forces(&mut self) -> (f64, f64, f64, f64, f64, f64) {
        let natoms = self.natoms();
        let apos = self.dyn_atoms.atoms.apos.as_slice();
        let fapos = self.dyn_atoms.fapos.as_mut_slice();
        let neighs = self.dyn_atoms.atoms.neighs.as_slice();
        let neigh_bs = self.dyn_atoms.atoms.neigh_bs.as_slice();
        let (eb, ea, ed, ei) = match self.bonded_mode {
            BondedFFMode::Uff => self.uff.eval_forces(apos, fapos, neighs, neigh_bs),
            BondedFFMode::RigidSp3 => (self.rigid_sp3.eval_forces(apos, fapos, &self.uff, neighs, neigh_bs), 0.0, 0.0, 0.0),
        };
        let mut enb = 0.0;
        let mut es = 0.0;

        // Non-bonded LJ + Coulomb (NonBondedFF owns REQs/PLQs)
        if let Some(ref mut nb) = self.nonbonded {
            enb = nb.eval(&mut fapos[0..natoms], &apos[0..natoms]);
        }

        // Surface interaction: borrows PLQH coefficients from NonBondedFF
        if let (Some(ref surf), Some(ref nb)) = (&self.surface, &self.nonbonded) {
            let plqs = nb.plqs.as_slice();
            es = surf.eval_all_clamped(&apos[0..natoms], &plqs[0..natoms], &mut fapos[0..natoms], 100.0);
        }

        (eb, ea, ed, ei, enb, es)
    }

    /// Single atom MD step. Returns (v·f, v·v, f·f) for convergence/instability checks.
    #[inline(always)]
    pub fn move_atom_md(&mut self, i: usize, dt: f64, flim: f64, cdamp: f64) -> (f64, f64, f64) {
        match self.bonded_mode {
            BondedFFMode::Uff => self.dyn_atoms.move_atom_md(i, dt, flim, cdamp),
            BondedFFMode::RigidSp3 => {
                let apos = self.dyn_atoms.atoms.apos.as_mut_slice();
                let fapos = self.dyn_atoms.fapos.as_slice();
                let vapos = self.dyn_atoms.vapos.as_mut_slice();
                let neigh_bs = self.dyn_atoms.atoms.neigh_bs.as_slice();
                self.rigid_sp3.move_atom_md(i, apos, fapos, vapos, &self.uff, neigh_bs, dt, flim, cdamp)
            }
        }
    }

    /// Run MD for niter steps or until force convergence.
    pub fn run_md(&mut self, niter: i32, dt: f64, fconv: f64, flim: f64, damping: f64) -> i32 {
        let f2conv = fconv * fconv;
        let cdamp = { let c = 1.0 - damping; if c < 0.0 { 0.0 } else { c } };
        for itr in 0..niter {
            let (eb, ea, ed, ei, enb, es) = self.eval_forces();
            let _etot = eb + ea + ed + ei + enb + es;
            let mut ff = 0.0;
            let mut vv = 0.0;
            let mut vf = 0.0;
            for ia in 0..self.natoms() {
                let (ff_, vv_, vf_) = self.move_atom_md(ia, dt, flim, cdamp);
                ff += ff_;
                vv += vv_;
                vf += vf_;
            }
            if ff < 0.0 {
                self.dyn_atoms.clean_velocity();
            }
            if vf < f2conv {
                return itr + 1;
            }
        }
        niter
    }

    // === Topology setup wrappers ===
    pub fn set_dummy_params(&mut self) { self.uff.set_dummy_params(self.dyn_atoms.apos()); }
    pub fn make_neigh_bs(&mut self) { self.dyn_atoms.atoms.make_neigh_bs(self.uff.bon_atoms.as_slice()); }
    pub fn bake_angle_neighs(&mut self) { self.uff.bake_angle_neighs(self.dyn_atoms.neighs()); }
    pub fn bake_dihedral_neighs(&mut self) { self.uff.bake_dihedral_neighs(self.dyn_atoms.neighs()); }
    pub fn bake_inversion_neighs(&mut self) { self.uff.bake_inversion_neighs(self.dyn_atoms.neighs()); }
    pub fn map_atom_interactions(&mut self) { self.uff.map_atom_interactions(); }
    pub fn update_hneigh(&mut self) { self.uff.update_hneigh(self.dyn_atoms.apos(), self.dyn_atoms.neighs()); }

    // === Convenience: attach surface ===
    pub fn setup_nacl_surface(&mut self, a: f64, z0: f64, beta_charge: f64, beta_morse_ratio: f64, q_amp: f64, plq_amp: f64) {
        self.surface = Some(surfff::setup_nacl_surface(a, z0, beta_charge, beta_morse_ratio, q_amp, plq_amp));
    }
}
