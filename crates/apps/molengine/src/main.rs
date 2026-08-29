use clap::Parser;
use rhai::{Engine, Dynamic, Array};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use surfmol::mol_world::MolWorld;
use surfmol::import::load_topology_from_json;

use molff::raff::{
    RaffState, RaffTopology, RaffConfig, NbConfig, NbParams, PortParam, BoxCfg,
    OrientMode, DynMode, PosSolver, FireState,
    step_force_md, step_inertial_reset, step_fire, step_position_based,
    solve_all_rotations, eval_port_forces, relax_position_based,
};

use numtypes::Vec3d;

/// Molecular simulation engine with Rhai scripting
#[derive(Parser, Debug)]
#[command(author, version, about = "Molecular forcefield simulation engine with Rhai scripting — UFF + RAFF")]
struct Args {
    /// Rhai script file to execute
    #[arg(short, long)]
    script: PathBuf,
}

/// RAFF solver mode (mirrors editor's RaffSolverMode enum).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RaffSolverMode { ForceMD, InertialReset, FIRE, PBD, XPBD, Projective }

impl RaffSolverMode {
    fn is_position_based(self) -> bool { matches!(self, Self::PBD | Self::XPBD | Self::Projective) }
    fn pos_solver(self) -> Option<PosSolver> {
        match self { Self::PBD => Some(PosSolver::PbdCompliance), Self::XPBD => Some(PosSolver::Xpbd), Self::Projective => Some(PosSolver::Projective), _ => None }
    }
}

/// Wrapper for MolWorld + optional RAFF state, thread-safe for Rhai.
#[derive(Clone)]
struct SimulationEngine {
    world: Arc<Mutex<MolWorld>>,
    // RAFF state (None until setup_raff is called)
    raff_state: Arc<Mutex<Option<RaffState>>>,
    raff_topo: Arc<Mutex<Option<RaffTopology>>>,
    raff_cfg: Arc<Mutex<RaffConfig>>,
    raff_nbcfg: Arc<Mutex<NbConfig>>,
    raff_solver: Arc<Mutex<RaffSolverMode>>,
    raff_fire: Arc<Mutex<Option<FireState>>>,
    // Cached elements (from load_topology) for non-bonded param setup
    elements: Arc<Mutex<Vec<String>>>,
}

impl SimulationEngine {
    fn new(world: MolWorld, elements: Vec<String>) -> Self {
        Self {
            world: Arc::new(Mutex::new(world)),
            raff_state: Arc::new(Mutex::new(None)),
            raff_topo: Arc::new(Mutex::new(None)),
            raff_cfg: Arc::new(Mutex::new(RaffConfig::default())),
            raff_nbcfg: Arc::new(Mutex::new(NbConfig::default())),
            raff_solver: Arc::new(Mutex::new(RaffSolverMode::ForceMD)),
            raff_fire: Arc::new(Mutex::new(None)),
            elements: Arc::new(Mutex::new(elements)),
        }
    }
}

/// Build RaffTopology + RaffState from the MolWorld's UFF bond topology + positions.
/// Mirrors editor's `build_raff_from_world`. Uses per-atom ARAP port geometry
/// (ports = initial neighbor directions → identity rotation = E_port=0).
fn build_raff_from_world(world: &MolWorld, elements: &[String]) -> (RaffState, RaffTopology) {
    let natoms = world.natoms();
    let mut topo = RaffTopology::new(natoms);
    // Bond params from UFF: [k, l0] → PortParam { k_p = k/2, l0 }
    let nbonds = world.uff.nbonds as usize;
    let bon_atoms = world.uff.bon_atoms.as_slice();
    let bon_params = world.uff.bon_params.as_slice();
    let mut bonds = Vec::with_capacity(nbonds);
    for ib in 0..nbonds {
        let ba = bon_atoms[ib];
        let bp = bon_params[ib];
        bonds.push([ba[0], ba[1]]);
        topo.bond_params.push(PortParam { k_p: bp[0] * 0.5, l0: bp[1] });
    }
    topo.build_neighs_from_bonds(&bonds);
    // State from current world positions
    let ps = world.dyn_atoms.atoms.apos.as_slice();
    topo.set_port_geometry_from_reference(&ps[0..natoms].to_vec());
    // Non-bonded params from element types: σ ≈ 2·rvdw, ε = 0.01 eV (weak default)
    for i in 0..natoms {
        let el = elements.get(i).map(|s| s.as_str()).unwrap_or("C");
        let rvdw = match el {
            "H" => 1.2, "C" => 1.7, "N" => 1.55, "O" => 1.52,
            "F" => 1.47, "Si" => 2.1, "P" => 1.8, "S" => 1.8,
            "Cl" => 1.75, _ => 1.7,
        };
        topo.nb_params[i] = NbParams { sigma: 2.0 * rvdw, epsilon: 0.01, charge: 0.0, radius: rvdw * 0.8 };
    }
    let mut state = RaffState::new(natoms);
    state.set_positions(&ps[0..natoms].to_vec());
    // Identity quaternions + initial adiabatic rotation solve
    for i in 0..natoms { state.quat[i] = numtypes::Quat4d::new(0.0, 0.0, 0.0, 1.0); }
    solve_all_rotations(&mut state, &topo);
    (state, topo)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut engine = Engine::new();
    engine.register_fn("println", |s: &str| { println!("{}", s); let _ = std::io::Write::flush(&mut std::io::stdout()); });

    // --- load_topology: returns SimulationEngine with UFF + elements cached ---
    engine.register_fn("load_topology", |path: &str| -> Dynamic {
        match load_topology_from_json(path) {
            Ok((ff, elements)) => {
                let world = MolWorld::from_uff(ff);
                Dynamic::from(SimulationEngine::new(world, elements))
            }
            Err(e) => { eprintln!("Error loading topology: {}", e); Dynamic::from(()) }
        }
    });

    // --- UFF API (existing) ---
    engine.register_fn("eval_forces", |sim: &mut SimulationEngine| -> f64 {
        let mut world = sim.world.lock().unwrap();
        let (eb, ea, ed, ei, enb, es) = world.eval_forces();
        eb + ea + ed + ei + enb + es
    });
    engine.register_fn("step_md", |sim: &mut SimulationEngine, dt: f64, flim: f64, damping: f64| {
        let mut world = sim.world.lock().unwrap();
        let cdamp = { let c = 1.0 - damping; if c < 0.0 { 0.0 } else { c } };
        for ia in 0..world.natoms() { world.move_atom_md(ia, dt, flim, cdamp); }
    });
    engine.register_fn("relax", |sim: &mut SimulationEngine, niter: i32, dt: f64, fconv: f64, flim: f64, damping: f64| -> i32 {
        let mut world = sim.world.lock().unwrap();
        world.run_md(niter, dt, fconv, flim, damping)
    });
    engine.register_fn("get_natoms", |sim: &mut SimulationEngine| -> i32 {
        sim.world.lock().unwrap().natoms() as i32
    });

    // --- setup_uff_params: load .dat files and fill UFF parameter arrays ---
    // Must be called before eval_forces/relax if topology was loaded from JSON
    // (JSON load creates Uff with zero params). data_dir should contain ElementTypes.dat,
    // BondTypes.dat, AngleTypes.dat, DihedralTypes.dat.
    engine.register_fn("setup_uff_params", |sim: &mut SimulationEngine, data_dir: &str| {
        use moltopo::params::Params;
        let mut world = sim.world.lock().unwrap();
        let mut params = Params::new();
        let dir = std::path::Path::new(data_dir);
        params.load_element_types(dir.join("ElementTypes.dat"));
        params.load_atom_types(dir.join("AtomTypes.dat"));
        params.load_bond_types(dir.join("BondTypes.dat"));
        params.load_angle_types(dir.join("AngleTypes.dat"));
        params.load_dihedral_types(dir.join("DihedralTypes.dat"));
        let types = sim.elements.lock().unwrap().clone();
        world.setup_uff_params(&params, &types);
        world.make_neigh_bs();
        world.bake_angle_neighs();
        world.bake_dihedral_neighs();
        world.bake_inversion_neighs();
        world.map_atom_interactions();
        world.update_hneigh();
        println!("[setup_uff_params] Loaded params from {} for {} atoms", data_dir, types.len());
    });

    // === RAFF API ===

    // --- setup_raff: build RaffTopology + RaffState from the loaded UFF topology ---
    engine.register_fn("setup_raff", |sim: &mut SimulationEngine| {
        let world = sim.world.lock().unwrap();
        let elements = sim.elements.lock().unwrap().clone();
        let (state, topo) = build_raff_from_world(&world, &elements);
        let natoms = topo.natoms;
        let nbonds = topo.bond_params.len();
        *sim.raff_state.lock().unwrap() = Some(state);
        *sim.raff_topo.lock().unwrap() = Some(topo);
        println!("[setup_raff] Built RAFF topology: {} atoms, {} bonds", natoms, nbonds);
    });

    // --- set_raff_solver: select solver mode ("forcemd"/"inertial"/"fire"/"pbd"/"xpbd"/"projective") ---
    engine.register_fn("set_raff_solver", |sim: &mut SimulationEngine, mode: &str| {
        let m = match mode {
            "forcemd" => RaffSolverMode::ForceMD,
            "inertial" => RaffSolverMode::InertialReset,
            "fire" => RaffSolverMode::FIRE,
            "pbd" => RaffSolverMode::PBD,
            "xpbd" => RaffSolverMode::XPBD,
            "projective" => RaffSolverMode::Projective,
            _ => { eprintln!("[set_raff_solver] unknown mode '{}', using forcemd", mode); RaffSolverMode::ForceMD }
        };
        *sim.raff_solver.lock().unwrap() = m;
        // Update cfg.dyn_mode + pos_solver for position-based solvers
        let mut cfg = sim.raff_cfg.lock().unwrap();
        if m.is_position_based() {
            cfg.dyn_mode = DynMode::Xpbd;
            cfg.pos_solver = m.pos_solver().unwrap();
        } else {
            cfg.dyn_mode = DynMode::ForceMD;
        }
        // Reset FIRE state when switching to/from FIRE
        if m == RaffSolverMode::FIRE {
            *sim.raff_fire.lock().unwrap() = Some(FireState::new(cfg.dt, cfg.dt * 10.0));
        } else {
            *sim.raff_fire.lock().unwrap() = None;
        }
    });

    // --- set_raff_orient: "adiabatic" or "dynamic" ---
    engine.register_fn("set_raff_orient", |sim: &mut SimulationEngine, mode: &str| {
        let mut cfg = sim.raff_cfg.lock().unwrap();
        cfg.orient_mode = match mode {
            "dynamic" => OrientMode::Dynamic,
            _ => OrientMode::Adiabatic,
        };
    });

    // --- set_raff_dt: set timestep ---
    engine.register_fn("set_raff_dt", |sim: &mut SimulationEngine, dt: f64| {
        sim.raff_cfg.lock().unwrap().dt = dt;
    });

    // --- set_raff_damping: set damping factor (0=kill, 1=no damping) ---
    engine.register_fn("set_raff_damping", |sim: &mut SimulationEngine, damping: f64| {
        let mut cfg = sim.raff_cfg.lock().unwrap();
        cfg.cdamp = damping;
        cfg.rot_damp = damping;
    });

    // --- set_raff_iters: set inner iterations for position-based solvers ---
    engine.register_fn("set_raff_iters", |sim: &mut SimulationEngine, iters: i64| {
        sim.raff_cfg.lock().unwrap().xpbd_iters = iters.max(1) as usize;
    });

    // --- set_raff_hb: set heavy-ball momentum (0 = disabled) ---
    engine.register_fn("set_raff_hb", |sim: &mut SimulationEngine, bmix: f64| {
        let mut cfg = sim.raff_cfg.lock().unwrap();
        cfg.bmix_end = bmix;
        if bmix > 0.0 { cfg.bmix_start = 0.0; cfg.bmix_istart = 3; cfg.bmix_iend = 10; }
    });

    // --- set_raff_pd_inertia: enable/disable PD outer-loop inertia ---
    engine.register_fn("set_raff_pd_inertia", |sim: &mut SimulationEngine, on: bool| {
        sim.raff_cfg.lock().unwrap().pd_inertia = on;
    });

    // --- set_raff_vel_reset: enable/disable velocity reset on v·F<0 ---
    engine.register_fn("set_raff_vel_reset", |sim: &mut SimulationEngine, on: bool| {
        sim.raff_cfg.lock().unwrap().vel_reset = on;
    });

    // --- set_raff_box: enable harmonic box constraint with min/max/k ---
    engine.register_fn("set_raff_box", |sim: &mut SimulationEngine, min_x: f64, min_y: f64, min_z: f64, max_x: f64, max_y: f64, max_z: f64, k: f64| {
        sim.raff_cfg.lock().unwrap().box_cfg = BoxCfg {
            enabled: true,
            min: Vec3d::new(min_x, min_y, min_z),
            max: Vec3d::new(max_x, max_y, max_z),
            k,
        };
    });

    // --- set_raff_nb: enable non-bonded with rcut/k_coll/f_max ---
    engine.register_fn("set_raff_nb", |sim: &mut SimulationEngine, enabled: bool, rcut: f64, k_coll: f64, f_max: f64| {
        let mut nbcfg = sim.raff_nbcfg.lock().unwrap();
        nbcfg.enabled = enabled;
        nbcfg.rcut = rcut;
        nbcfg.k_coll = k_coll;
        nbcfg.f_max = f_max;
    });

    // --- set_raff_charges: set per-atom charges from a Rhai array ---
    engine.register_fn("set_raff_charges", |sim: &mut SimulationEngine, charges: Array| {
        if let Some(topo) = sim.raff_topo.lock().unwrap().as_mut() {
            for (i, c) in charges.iter().enumerate() {
                if i < topo.natoms {
                    if let Ok(q) = c.as_float() { topo.nb_params[i].charge = q; }
                }
            }
        }
    });

    // --- raff_step: one RAFF relaxation step (returns total energy) ---
    engine.register_fn("raff_step", |sim: &mut SimulationEngine| -> f64 {
        let solver = *sim.raff_solver.lock().unwrap();
        let cfg = *sim.raff_cfg.lock().unwrap();
        let nbcfg = *sim.raff_nbcfg.lock().unwrap();
        let topo_guard = sim.raff_topo.lock().unwrap();
        let mut state_guard = sim.raff_state.lock().unwrap();
        let topo = topo_guard.as_ref().expect("raff_step: call setup_raff first");
        let state = state_guard.as_mut().expect("raff_step: call setup_raff first");
        let np = topo.natoms;
        let mut fapos = vec![Vec3d::new(0.0, 0.0, 0.0); np];
        let mut tau = vec![Vec3d::new(0.0, 0.0, 0.0); np];
        let e = match solver {
            RaffSolverMode::ForceMD => step_force_md(state, topo, &cfg, &mut fapos, &mut tau, &nbcfg).0,
            RaffSolverMode::InertialReset => step_inertial_reset(state, topo, &cfg, &mut fapos, &mut tau, &nbcfg).0,
            RaffSolverMode::FIRE => {
                let mut fire_guard = sim.raff_fire.lock().unwrap();
                let fire = fire_guard.get_or_insert_with(|| FireState::new(cfg.dt, cfg.dt * 10.0));
                step_fire(state, topo, &cfg, fire, &mut fapos, &mut tau, &nbcfg).0
            }
            RaffSolverMode::PBD | RaffSolverMode::XPBD | RaffSolverMode::Projective => {
                step_position_based(state, topo, &cfg, &nbcfg)
            }
        };
        e
    });

    // --- raff_relax: run N RAFF steps, print progress, return (final_energy, n_steps, converged) ---
    // Returns an array [energy, n_steps, converged(0/1)] for Rhai.
    engine.register_fn("raff_relax", |sim: &mut SimulationEngine, max_steps: i64, e_tol: f64| -> Array {
        let solver = *sim.raff_solver.lock().unwrap();
        let cfg = *sim.raff_cfg.lock().unwrap();
        let nbcfg = *sim.raff_nbcfg.lock().unwrap();
        let topo_guard = sim.raff_topo.lock().unwrap();
        let mut state_guard = sim.raff_state.lock().unwrap();
        let topo = topo_guard.as_ref().expect("raff_relax: call setup_raff first");
        let state = state_guard.as_mut().expect("raff_relax: call setup_raff first");
        let np = topo.natoms;
        let max_steps = max_steps.max(0) as usize;
        let (e, n, converged) = if solver.is_position_based() {
            let (e, n, conv, _n_evals) = relax_position_based(state, topo, &cfg, &nbcfg, max_steps, e_tol);
            (e, n, conv)
        } else {
            // ForceMD / InertialReset / FIRE — manual relax loop with energy + force convergence
            let mut fapos = vec![Vec3d::new(0.0, 0.0, 0.0); np];
            let mut tau = vec![Vec3d::new(0.0, 0.0, 0.0); np];
            let mut last_e = f64::INFINITY;
            let mut fire_state = if solver == RaffSolverMode::FIRE {
                Some(FireState::new(cfg.dt, cfg.dt * 10.0))
            } else { None };
            let mut n_done = 0usize;
            let mut conv = false;
            for step in 0..max_steps {
                let (e, max_f, _max_t) = match solver {
                    RaffSolverMode::ForceMD => step_force_md(state, topo, &cfg, &mut fapos, &mut tau, &nbcfg),
                    RaffSolverMode::InertialReset => step_inertial_reset(state, topo, &cfg, &mut fapos, &mut tau, &nbcfg),
                    RaffSolverMode::FIRE => {
                        let fire = fire_state.as_mut().unwrap();
                        step_fire(state, topo, &cfg, fire, &mut fapos, &mut tau, &nbcfg)
                    }
                    _ => unreachable!(),
                };
                if step % 100 == 0 {
                    println!("[raff_relax {:?}] step {} E={:.6} max|F|={:.6}", solver, step, e, max_f);
                }
                n_done = step + 1;
                if (last_e - e).abs() < e_tol && step > 10 { conv = true; break; }
                if max_f < e_tol && step > 10 { conv = true; break; }
                last_e = e;
            }
            (last_e, n_done, conv)
        };
        println!("[raff_relax] done: E={:.6}, {} steps, converged={}", e, n, converged);
        let mut arr = Array::new();
        arr.push(e.into());
        arr.push((n as i64).into());
        arr.push((if converged { 1 } else { 0 }).into());
        arr
    });

    // --- get_raff_energy: evaluate port + non-bonded energy without stepping ---
    engine.register_fn("get_raff_energy", |sim: &mut SimulationEngine| -> f64 {
        let topo_guard = sim.raff_topo.lock().unwrap();
        let mut state_guard = sim.raff_state.lock().unwrap();
        let nbcfg = *sim.raff_nbcfg.lock().unwrap();
        let topo = topo_guard.as_ref().expect("get_raff_energy: call setup_raff first");
        let state = state_guard.as_mut().expect("get_raff_energy: call setup_raff first");
        let np = topo.natoms;
        let mut fapos = vec![Vec3d::new(0.0, 0.0, 0.0); np];
        let mut tau = vec![Vec3d::new(0.0, 0.0, 0.0); np];
        let e_port = eval_port_forces(state, topo, &mut fapos, &mut tau);
        let e_nb = molff::raff::eval_nonbonded(state, topo, &nbcfg, &mut fapos);
        e_port + e_nb
    });

    // --- get_raff_pos: return positions as flat array [x0,y0,z0, x1,y1,z1, ...] ---
    engine.register_fn("get_raff_pos", |sim: &mut SimulationEngine| -> Array {
        let state_guard = sim.raff_state.lock().unwrap();
        let state = state_guard.as_ref().expect("get_raff_pos: call setup_raff first");
        let mut arr = Array::new();
        for p in &state.pos { arr.push(p.x.into()); arr.push(p.y.into()); arr.push(p.z.into()); }
        arr
    });

    // --- save_raff_xyz: write current RAFF positions to an XYZ file ---
    engine.register_fn("save_raff_xyz", |sim: &mut SimulationEngine, path: &str| {
        let state_guard = sim.raff_state.lock().unwrap();
        let elements = sim.elements.lock().unwrap();
        let state = state_guard.as_ref().expect("save_raff_xyz: call setup_raff first");
        let mut s = format!("{}\nRAFF relaxed structure\n", state.natoms);
        for i in 0..state.natoms {
            let el = elements.get(i).map(|s| s.as_str()).unwrap_or("C");
            let p = state.pos[i];
            s.push_str(&format!("{} {:.6} {:.6} {:.6}\n", el, p.x, p.y, p.z));
        }
        std::fs::write(path, s).expect("save_raff_xyz: write failed");
        println!("[save_raff_xyz] Wrote {} atoms to {}", state.natoms, path);
    });

    // Read and execute the script
    let script = std::fs::read_to_string(&args.script)
        .map_err(|e| format!("Failed to read script file {:?}: {}", args.script, e))?;
    engine.run(&script)?;
    Ok(())
}
