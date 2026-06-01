// Example Rhai script for molecular dynamics simulation
// Usage: mol_engine --script md.rhai

print("Loading topology from topology.json...");
let sim = load_topology("topology.json");

print("Number of atoms: " + get_natoms(sim));

print("Starting MD simulation...");
let nsteps = 10000;
let dt = 0.01;
let flim = 1000.0;
let damping = 0.05;

for i in 0..nsteps {
    step_md(sim, dt, flim, damping);
    
    if i % 100 == 0 {
        let e = eval_forces(sim);
        print("Step " + i + " Energy: " + e);
    }
}

print("MD simulation complete");
print("Final energy: " + eval_forces(sim));
