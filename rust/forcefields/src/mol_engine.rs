use clap::Parser;
use rhai::{Engine, Dynamic};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use surfmol_forcefields::mol_world::MolWorld;
use surfmol_forcefields::import::load_topology_from_json;

/// Molecular simulation engine with Rhai scripting
#[derive(Parser, Debug)]
#[command(author, version, about = "Molecular forcefield simulation engine with Rhai scripting")]
struct Args {
    /// Rhai script file to execute
    #[arg(short, long)]
    script: PathBuf,
}

/// Wrapper for MolWorld to make it thread-safe for Rhai
#[derive(Clone)]
struct SimulationEngine {
    world: Arc<Mutex<MolWorld>>,
}

impl SimulationEngine {
    fn new(world: MolWorld) -> Self {
        Self {
            world: Arc::new(Mutex::new(world)),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize Rhai engine
    let mut engine = Engine::new();

    // Register basic functions
    engine.register_fn("print", |s: &str| println!("{}", s));

    // Register topology loading function - returns a Dynamic containing the engine
    engine.register_fn("load_topology", |path: &str| -> Dynamic {
        match load_topology_from_json(path) {
            Ok((ff, _elements)) => {
                let world = MolWorld::from_uff(ff);
                Dynamic::from(SimulationEngine::new(world))
            }
            Err(e) => {
                eprintln!("Error loading topology: {}", e);
                Dynamic::from(())
            }
        }
    });

    // Register UFF engine methods
    engine.register_fn("eval_forces", |sim: &mut SimulationEngine| -> f64 {
        let mut world = sim.world.lock().unwrap();
        let (eb, ea, ed, ei, enb, es) = world.eval_forces();
        eb + ea + ed + ei + enb + es
    });

    engine.register_fn("step_md", |sim: &mut SimulationEngine, dt: f64, flim: f64, damping: f64| {
        let mut world = sim.world.lock().unwrap();
        let cdamp = {
            let c = 1.0 - damping;
            if c < 0.0 { 0.0 } else { c }
        };
        for ia in 0..world.natoms() {
            world.move_atom_md(ia, dt, flim, cdamp);
        }
    });

    engine.register_fn("relax", |sim: &mut SimulationEngine, niter: i32, dt: f64, fconv: f64, flim: f64, damping: f64| -> i32 {
        let mut world = sim.world.lock().unwrap();
        world.run_md(niter, dt, fconv, flim, damping)
    });

    engine.register_fn("get_natoms", |sim: &mut SimulationEngine| -> i32 {
        let world = sim.world.lock().unwrap();
        world.natoms() as i32
    });

    // Read and execute the script
    let script = std::fs::read_to_string(&args.script)
        .map_err(|e| format!("Failed to read script file {:?}: {}", args.script, e))?;

    engine.run(&script)?;

    Ok(())
}
