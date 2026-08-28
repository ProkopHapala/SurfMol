//! Diagnostic test: verify RAFF per-atom ARAP port geometry works for benzene.
//! Regression test for the bug where idealized sp2 ports (set_port_geometry_from_types)
//! produced geometrically inconsistent port-to-neighbor assignments, causing huge forces
//! and H atoms collapsing into the ring center.
//! The fix: set_port_geometry_from_reference uses initial neighbor directions (ARAP).

use molff::raff::*;
use numtypes::{Vec3d, Quat4d};

#[test]
fn test_benzene_diag() {
    // Perfect planar benzene: 6 C in hexagon (radius 1.4 Å), 6 H outward (radius 2.49 Å)
    let rc = 1.4;  // C-C distance ~1.4 Å
    let rcc = rc;  // circumradius = bond length for hexagon
    let rch = 1.09; // C-H bond length
    let rh = rcc + rch; // H distance from center

    let mut pos = vec![Vec3d::new(0.0,0.0,0.0); 12];
    for i in 0..6 {
        let a = i as f64 * std::f64::consts::PI / 3.0;
        pos[i]   = Vec3d::new(rcc * a.cos(), rcc * a.sin(), 0.0);  // C at hexagon vertex
        pos[6+i] = Vec3d::new(rh  * a.cos(), rh  * a.sin(), 0.0);  // H outward
    }

    println!("=== Benzene positions (planar) ===");
    for i in 0..12 {
        let el = if i < 6 { "C" } else { "H" };
        println!("atom {:2} {}  pos=({:8.4}, {:8.4}, {:8.4})", i, el, pos[i].x, pos[i].y, pos[i].z);
    }

    // Bonds: C0-C1, C1-C2, C2-C3, C3-C4, C4-C5, C5-C0, C0-H6, C1-H7, ... C5-H11
    let bonds: Vec<[i32; 2]> = vec![
        [0,1], [1,2], [2,3], [3,4], [4,5], [5,0],  // ring
        [0,6], [1,7], [2,8], [3,9], [4,10], [5,11], // C-H
    ];

    // UFF types
    let uff_types: Vec<String> = (0..6).map(|_| "C_R".to_string())
        .chain((0..6).map(|_| "H_".to_string()))
        .collect();

    // Build RAFF topology
    let natoms = 12;
    let mut topo = RaffTopology::new(natoms);
    for ib in 0..bonds.len() {
        topo.bond_params.push(PortParam { k_p: 250.0, l0: if ib < 6 { 1.40 } else { 1.09 } });
    }
    topo.build_neighs_from_bonds(&bonds);
    // Use per-atom ARAP: ports from initial neighbor directions
    topo.set_port_geometry_from_reference(&pos);

    println!("\n=== Port assignments ===");
    for i in 0..12 {
        let np = topo.nport[i] as usize;
        let ns = topo.neighs[i].as_array();
        let bs = topo.neigh_bs[i].as_array();
        let el = if i < 6 { "C" } else { "H" };
        println!("atom {:2} {} nport={}", i, el, np);
        for s in 0..np {
            let j = ns[s] as usize;
            let d = pos[j] - pos[i];
            let d_norm = d.norm();
            let port = topo.port_local[i*4 + s];
            println!("  port {}: neighbor={:2} d=({:7.4},{:7.4},{:7.4}) |d|={:.4}  port_local=({:7.4},{:7.4},{:7.4})",
                s, j, d.x, d.y, d.z, d_norm, port.x, port.y, port.z);
        }
    }

    // Build state
    let mut state = RaffState::new(natoms);
    state.set_positions(&pos);
    for i in 0..natoms { state.quat[i] = Quat4d::new(0.0, 0.0, 0.0, 1.0); }

    // Solve initial rotations
    solve_all_rotations(&mut state, &topo);

    println!("\n=== Rotations after solve_all_rotations ===");
    for i in 0..6 {
        let q = state.quat[i];
        println!("C{:2} quat=({:9.6}, {:9.6}, {:9.6}, {:9.6})", i, q.x, q.y, q.z, q.w);
        // Check port tips vs neighbor positions
        let np = topo.nport[i] as usize;
        let ns = topo.neighs[i].as_array();
        for s in 0..np {
            let j = ns[s] as usize;
            let tip = topo.port_tip(&state, i, s);
            let err = (pos[j] - tip).norm();
            println!("  port {} -> neighbor {:2}: tip=({:7.4},{:7.4},{:7.4})  neighbor=({:7.4},{:7.4},{:7.4})  err={:.6}",
                s, j, tip.x, tip.y, tip.z, pos[j].x, pos[j].y, pos[j].z, err);
        }
    }

    // Evaluate forces
    let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); natoms];
    let mut tau = vec![Vec3d::new(0.0,0.0,0.0); natoms];
    let e_port = eval_port_forces(&state, &topo, &mut fapos, &mut tau);

    println!("\n=== Port forces (initial) ===");
    println!("E_port = {:.6}", e_port);
    for i in 0..12 {
        let el = if i < 6 { "C" } else { "H" };
        let f = fapos[i];
        let t = tau[i];
        let fmag = f.norm();
        println!("atom {:2} {} F=({:9.4},{:9.4},{:9.4}) |F|={:.6}  tau=({:9.4},{:9.4},{:9.4})",
            i, el, f.x, f.y, f.z, fmag, t.x, t.y, t.z);
    }

    // Run 10 relaxation steps and print H positions
    println!("\n=== Relaxation trace (10 steps, dt=0.02, damping=0.1) ===");
    let dt = 0.02;
    let cdamp = 0.9;
    let rot_damp = 0.9;
    for step in 0..10 {
        // Eval forces
        let mut fapos = vec![Vec3d::new(0.0,0.0,0.0); natoms];
        let mut tau = vec![Vec3d::new(0.0,0.0,0.0); natoms];
        let e_port = eval_port_forces(&state, &topo, &mut fapos, &mut tau);

        // 2D constraint
        for i in 0..natoms { fapos[i].z = 0.0; tau[i].x = 0.0; tau[i].y = 0.0; }

        // Integrate
        for i in 0..natoms {
            let mut f = fapos[i];
            let f2 = f.norm2();
            if f2 > 100.0*100.0 { f.mul(100.0 / f2.sqrt()); }
            let mut v = state.vel[i];
            v.mul(cdamp);
            v.add_mul(f, dt / topo.mass[i]);
            state.vel[i] = v;
            state.pos[i].add_mul(v, dt);
            // Rotation
            if topo.inv_inertia[i] > 0.0 && topo.nport[i] > 0 {
                let t = tau[i];
                let mut w = state.omega[i];
                w.mul(rot_damp);
                w.add_mul(t, topo.inv_inertia[i] * dt);
                state.omega[i] = w;
                let dq = quat_from_omega_dt(w, dt);
                state.quat[i] = quat_normalize(quat_mul(dq, state.quat[i]));
            }
        }
        // Adiabatic: re-solve rotations
        solve_all_rotations(&mut state, &topo);
        // 2D clamp
        for i in 0..natoms { state.pos[i].z = 0.0; state.vel[i].z = 0.0; }

        // Print H positions
        print!("step {:2} E={:.6}  H pos: ", step, e_port);
        for i in 6..12 {
            let r = state.pos[i].norm();
            let a = state.pos[i].y.atan2(state.pos[i].x);
            print!("H{} r={:.3} a={:.1}°  ", i-6, r, a.to_degrees());
        }
        println!();
    }

    println!("\n=== Final C positions ===");
    for i in 0..6 {
        let r = state.pos[i].norm();
        println!("C{} pos=({:.4},{:.4},{:.4}) r={:.4}", i, state.pos[i].x, state.pos[i].y, state.pos[i].z, r);
    }

    // Regression assertions: per-atom ARAP should give E_port=0 and stable structure
    assert!(e_port < 1e-10, "E_port should be 0 at init with per-atom ARAP, got {}", e_port);
    // After 10 steps with no external forces, atoms should not have moved significantly
    for i in 0..12 {
        let dr = (state.pos[i] - pos[i]).norm();
        assert!(dr < 1e-6, "atom {} moved {} — should be stationary with no external forces", i, dr);
    }
    // H atoms should stay outside the ring (r > 2.0)
    for i in 6..12 {
        let r = state.pos[i].norm();
        assert!(r > 2.0, "H{} at r={:.3} — should stay outside ring (r>2.0)", i-6, r);
    }
}
