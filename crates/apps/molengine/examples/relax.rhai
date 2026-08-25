// Example Rhai script for geometry relaxation
// Usage: mol_engine --script relax.rhai

print("Loading topology from topology.json...");
let sim = load_topology("topology.json");

print("Number of atoms: " + get_natoms(sim));

print("Starting relaxation...");
let niter = 1000;
let dt = 0.01;
let fconv = 1e-3;
let flim = 1000.0;
let damping = 0.05;

let converged = relax(sim, niter, dt, fconv, flim, damping);

if converged < niter {
    print("Converged in " + converged + " steps");
} else {
    print("Did not converge in " + niter + " steps");
}

print("Final energy: " + eval_forces(sim));
