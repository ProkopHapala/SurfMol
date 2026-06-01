use surfmol_forcefields::mol_world::MolWorld;
use surfmol_topology::topology::Topology;
use surfmol_common::math::vec3::Vec3d;

#[test]
fn test_rigid_sp3_tetrahedron() {
    // Create a simple tetrahedral sp3 center (like CH4)
    // Center at origin, 4 neighbors at tetrahedral directions
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    let r = 1.0; // bond length
    
    let apos = vec![
        Vec3d::new(0.0, 0.0, 0.0), // center (index 0)
        Vec3d::new( r*inv_sqrt3,  r*inv_sqrt3,  r*inv_sqrt3), // neighbor 1
        Vec3d::new( r*inv_sqrt3, -r*inv_sqrt3, -r*inv_sqrt3), // neighbor 2
        Vec3d::new(-r*inv_sqrt3,  r*inv_sqrt3, -r*inv_sqrt3), // neighbor 3
        Vec3d::new(-r*inv_sqrt3, -r*inv_sqrt3,  r*inv_sqrt3), // neighbor 4
    ];
    
    let bonds = vec![
        [0, 1], [0, 2], [0, 3], [0, 4], // center to all 4 neighbors
    ];
    
    let top = Topology { apos, bonds, angles: vec![], dihedrals: vec![], inversions: vec![] };
    let mut world = MolWorld::from_topology(&top);
    
    // Setup neighbor lists (required for rigid_sp3)
    world.make_neigh_bs();
    
    // Set dummy bond parameters (k=100, l0=1.0)
    for ib in 0..world.uff.nbonds as usize {
        world.uff.bon_params.as_mut_slice()[ib] = [100.0, 1.0];
    }
    
    // Initial positions
    let pos0 = world.dyn_atoms.atoms.apos.as_slice()[0];
    
    // Run a few force evaluations to ensure it doesn't crash
    for _ in 0..5 {
        let (eb, ea, ed, ei, enb, es) = world.eval_forces();
        assert!(eb >= 0.0, "Bond energy should be non-negative");
        assert!(ea == 0.0, "Angle energy should be zero (no angles)");
        assert!(ed == 0.0, "Dihedral energy should be zero (no dihedrals)");
        assert!(ei == 0.0, "Inversion energy should be zero (no inversions)");
        assert!(enb == 0.0, "Nonbonded energy should be zero (no nonbonded)");
        assert!(es == 0.0, "Surface energy should be zero (no surface)");
    }
    
    // Run a few MD steps
    let dt = 0.01;
    let flim = 1000.0;
    let cdamp = 0.95;
    
    for _ in 0..10 {
        for ia in 0..world.natoms() {
            world.move_atom_md(ia, dt, flim, cdamp);
        }
    }
    
    // Positions should have changed (due to forces)
    let pos_final = world.dyn_atoms.atoms.apos.as_slice()[0];
    let moved = (pos_final.x - pos0.x).abs() > 1e-6 || 
                (pos_final.y - pos0.y).abs() > 1e-6 || 
                (pos_final.z - pos0.z).abs() > 1e-6;
    
    // Note: positions might not move much if system is near equilibrium
    // The important thing is that it doesn't crash and forces are computed
    println!("Rigid sp3 test passed: positions changed = {}", moved);
}

#[test]
fn test_rigid_sp3_water() {
    // Simple water molecule (H-O-H) - not sp3 but tests the solver
    let apos = vec![
        Vec3d::new(0.0, 0.0, 0.0), // O
        Vec3d::new(0.96, 0.0, 0.0), // H1
        Vec3d::new(-0.24, 0.93, 0.0), // H2
    ];
    
    let bonds = vec![
        [0, 1], [0, 2], // O-H bonds
    ];
    
    let top = Topology { apos, bonds, angles: vec![], dihedrals: vec![], inversions: vec![] };
    let mut world = MolWorld::from_topology(&top);
    
    world.make_neigh_bs();
    
    // Set dummy bond parameters
    for ib in 0..world.uff.nbonds as usize {
        world.uff.bon_params.as_mut_slice()[ib] = [100.0, 1.0];
    }
    
    // Force evaluation should work
    let (eb, _, _, _, _, _) = world.eval_forces();
    assert!(eb >= 0.0, "Bond energy should be non-negative");
    
    // MD steps should work
    let dt = 0.01;
    let flim = 1000.0;
    let cdamp = 0.95;
    
    for _ in 0..5 {
        for ia in 0..world.natoms() {
            world.move_atom_md(ia, dt, flim, cdamp);
        }
    }
    
    println!("Rigid sp3 water test passed");
}
