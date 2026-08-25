use std::fs::File;
use std::io::Write;
use numcore::math::vec3::Vec3d;
use surfff::{setup_nacl_surface, SurfaceFolded};

/// Simple line-plot SVG generator (pure Rust, no external plotting deps).
fn plot_svg(title: &str, xlabel: &str, ylabel: &str, w: u32, h: u32,
            series: &[(String, Vec<(f64,f64)>, String)]) -> String {
    let margin = 60.0;
    let pw = (w as f64 - 2.0*margin);
    let ph = (h as f64 - 2.0*margin);
    let mut all_x: Vec<f64> = series.iter().flat_map(|(_, pts, _)| pts.iter().map(|p| p.0)).collect();
    let mut all_y: Vec<f64> = series.iter().flat_map(|(_, pts, _)| pts.iter().map(|p| p.1)).collect();
    all_x.sort_by(|a,b| a.partial_cmp(b).unwrap());
    all_y.sort_by(|a,b| a.partial_cmp(b).unwrap());
    let x_min = *all_x.first().unwrap_or(&0.0);
    let x_max = *all_x.last().unwrap_or(&1.0);
    let y_min = *all_y.first().unwrap_or(&0.0);
    let y_max = *all_y.last().unwrap_or(&1.0);
    let x_range = (x_max - x_min).max(1e-12);
    let y_range = (y_max - y_min).max(1e-12);
    let sx = |x: f64| margin + (x - x_min) / x_range * pw;
    let sy = |y: f64| (h as f64 - margin) - (y - y_min) / y_range * ph;

    let mut svg = format!(r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" style="background:{}">
  <text x="{}" y="25" text-anchor="middle" font-size="16" font-family="sans-serif">{}</text>
  <text x="{}" y="{}" text-anchor="middle" font-size="13" font-family="sans-serif">{}</text>
  <text x="15" y="{}" text-anchor="middle" font-size="13" font-family="sans-serif" transform="rotate(-90 15 {})">{}</text>
"#, w, h, "#fff", w as f64 / 2.0, title, w as f64 / 2.0, h as f64 - 15.0, xlabel, h as f64 / 2.0, h as f64 / 2.0, ylabel);

    // Grid lines
    for i in 0..=5 {
        let fx = i as f64 / 5.0;
        let xg = sx(x_min + fx * x_range);
        let yg = sy(y_min + fx * y_range);
        svg += &format!(r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#, xg, margin, xg, h as f64 - margin, "#ddd");
        svg += &format!(r#"  <text x="{}" y="{}" text-anchor="middle" font-size="10" fill="{}">{:.2}</text>
"#, xg, h as f64 - margin + 15.0, "#666", x_min + fx * x_range);
        svg += &format!(r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>
"#, margin, yg, w as f64 - margin, yg, "#ddd");
        svg += &format!(r#"  <text x="{}" y="{}" text-anchor="end" font-size="10" fill="{}">{:.2}</text>
"#, margin - 5.0, yg + 4.0, "#666", y_min + fx * y_range);
    }

    // Axes
    svg += &format!(r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1.5"/>
  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1.5"/>
"#, margin, h as f64 - margin, w as f64 - margin, h as f64 - margin, "#333", margin, margin, margin, h as f64 - margin, "#333");

    // Series
    let colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"];
    for (idx, (name, pts, _)) in series.iter().enumerate() {
        let c = colors[idx % colors.len()];
        if pts.len() < 2 { continue; }
        let mut d = format!("M {:.2} {:.2}", sx(pts[0].0), sy(pts[0].1));
        for i in 1..pts.len() { d += &format!(" L {:.2} {:.2}", sx(pts[i].0), sy(pts[i].1)); }
        svg += &format!(r#"  <path d="{}" fill="none" stroke="{}" stroke-width="1.5"/>
"#, d, c);
        // Legend
        let lx = w as f64 - margin - 120.0;
        let ly = margin + 20.0 + (idx * 18) as f64;
        svg += &format!(r#"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>
  <text x="{}" y="{}" font-size="11" font-family="sans-serif">{}</text>
"#, lx, ly, lx + 15.0, ly, c, lx + 20.0, ly + 4.0, name);
    }
    svg += "</svg>";
    svg
}

fn main() {
    // NaCl surface parameters
    let a = 5.66;        // lattice constant [Å] (NaCl conventional cell)
    let z0 = 0.0;         // surface plane z
    let beta_charge = 0.3;   // electrostatics z-decay (slower)
    let beta_morse_ratio = 2.0; // Morse decay = ratio * charge decay
    let q_amp = 1.0;      // electrostatic amplitude
    let plq_amp = 1.0;    // Pauli/London amplitude

    let surf = setup_nacl_surface(a, z0, beta_charge, beta_morse_ratio, q_amp, plq_amp);
    let beta_morse = beta_charge * beta_morse_ratio;
    println!("=== Surface z-basis decays ===");
    println!("  Pauli (repulsive):  kz = {:.2}  (2*beta_morse)", 2.0 * beta_morse);
    println!("  London (attractive): kz = {:.2}  (beta_morse)", beta_morse);
    println!("  Charge (electrostatics): kz = {:.2}  (beta_charge)", beta_charge);

    // Test atom: Carbon-like (RvdW=1.7Å, EvdW=0.1eV)
    let alpha = 2.0;
    let req_neutral  = [1.7, 0.1, 0.0,  0.0];
    let req_positive = [1.7, 0.1, 0.5,  0.0];
    let req_negative = [1.7, 0.1, -0.5, 0.0];

    let plq_n = SurfaceFolded::req2plq(req_neutral,  alpha);
    let plq_p = SurfaceFolded::req2plq(req_positive, alpha);
    let plq_m = SurfaceFolded::req2plq(req_negative, alpha);

    println!("\n=== Atom PLQ (alpha={}) ===", alpha);
    println!("  neutral:  Pauli={:.6}  London={:.6}  Q={:.6}", plq_n[0], plq_n[1], plq_n[2]);

    // --- Scan 1: Vary z at (x=0, y=0) --- NEUTRAL ONLY for vdW shape clarity
    let z_min = -2.0;
    let z_max = 10.0;
    let nz = 120;
    let dz = (z_max - z_min) / (nz as f64);

    let mut z_pts: Vec<(f64,f64)> = Vec::with_capacity(nz+1);
    let mut fz_pts: Vec<(f64,f64)> = Vec::with_capacity(nz+1);

    {
        let mut file = File::create("surface_z_scan.csv").unwrap();
        writeln!(file, "z,E,Fx,Fy,Fz").unwrap();
        for iz in 0..=nz {
            let z = z_min + iz as f64 * dz;
            let pos = Vec3d::new(0.0, 0.0, z);
            let (e, f) = surf.eval_atom(pos, plq_n);
            writeln!(file, "{:.4},{:.6},{:.6},{:.6},{:.6}", z, e, f.x, f.y, f.z).unwrap();
            z_pts.push((z, e));
            fz_pts.push((z, f.z));
        }
    }
    println!("\nWrote surface_z_scan.csv (neutral atom, z={:.1}..{:.1})", z_min, z_max);

    // --- Scan 2: Vary x at fixed z=2.0Å --- NEUTRAL ONLY
    let x_min = 0.0;
    let x_max = a * 2.0;
    let nx_step = 100;
    let dx_step = (x_max - x_min) / (nx_step as f64);
    let z_fixed = 2.0;

    let mut x_pts: Vec<(f64,f64)> = Vec::with_capacity(nx_step+1);
    let mut fx_pts: Vec<(f64,f64)> = Vec::with_capacity(nx_step+1);

    {
        let mut file = File::create("surface_x_scan.csv").unwrap();
        writeln!(file, "x,E,Fx,Fy,Fz").unwrap();
        for ix in 0..=nx_step {
            let x = x_min + ix as f64 * dx_step;
            let pos = Vec3d::new(x, 0.0, z_fixed);
            let (e, f) = surf.eval_atom(pos, plq_n);
            writeln!(file, "{:.4},{:.6},{:.6},{:.6},{:.6}", x, e, f.x, f.y, f.z).unwrap();
            x_pts.push((x, e));
            fx_pts.push((x, f.x));
        }
    }
    println!("Wrote surface_x_scan.csv (neutral atom, x={:.1}..{:.1}, z={:.1})", x_min, x_max, z_fixed);

    // --- Spot checks: all three charge states to verify electrostatics ---
    println!("\n=== Spot checks (neutral / +0.5e / -0.5e) ===");
    let spots = [
        ("Na (0,0,2)",       Vec3d::new(0.0,       0.0,       2.0)),
        ("Cl (a/2,0,2)",     Vec3d::new(a * 0.5,   0.0,       2.0)),
        ("Na (a/2,a/2,2)",   Vec3d::new(a * 0.5,   a * 0.5,   2.0)),
        ("bridge (a/4,a/4,2)", Vec3d::new(a * 0.25, a * 0.25, 2.0)),
        ("penetrate (0,0,-1)", Vec3d::new(0.0,     0.0,      -1.0)),
    ];
    for (name, pos) in &spots {
        let (e_n, f_n) = surf.eval_atom(*pos, plq_n);
        let (e_p, f_p) = surf.eval_atom(*pos, plq_p);
        let (e_m, f_m) = surf.eval_atom(*pos, plq_m);
        println!("\n  {}:", name);
        println!("    neutral:  E={:10.4} eV  Fz={:8.4}", e_n, f_n.z);
        println!("    positive: E={:10.4} eV  Fz={:8.4}", e_p, f_p.z);
        println!("    negative: E={:10.4} eV  Fz={:8.4}", e_m, f_m.z);
    }

    // --- Generate SVG plots: NEUTRAL ONLY ---
    let plot_e_z = plot_svg("Neutral atom: E vs z", "z [Å]", "E [eV]", 700, 400,
        &[(String::from("E(z)"), z_pts, String::from("#e41a1c"))]);
    File::create("plot_E_vs_z.svg").unwrap().write_all(plot_e_z.as_bytes()).unwrap();
    println!("\nWrote plot_E_vs_z.svg");

    let plot_fz_z = plot_svg("Neutral atom: Fz vs z", "z [Å]", "Fz [eV/Å]", 700, 400,
        &[(String::from("Fz(z)"), fz_pts, String::from("#377eb8"))]);
    File::create("plot_Fz_vs_z.svg").unwrap().write_all(plot_fz_z.as_bytes()).unwrap();
    println!("Wrote plot_Fz_vs_z.svg");

    let plot_e_x = plot_svg("Neutral atom: E vs x (z=2Å)", "x [Å]", "E [eV]", 700, 400,
        &[(String::from("E(x)"), x_pts, String::from("#4daf4a"))]);
    File::create("plot_E_vs_x.svg").unwrap().write_all(plot_e_x.as_bytes()).unwrap();
    println!("Wrote plot_E_vs_x.svg");

    let plot_fx_x = plot_svg("Neutral atom: Fx vs x (z=2Å)", "x [Å]", "Fx [eV/Å]", 700, 400,
        &[(String::from("Fx(x)"), fx_pts, String::from("#984ea3"))]);
    File::create("plot_Fx_vs_x.svg").unwrap().write_all(plot_fx_x.as_bytes()).unwrap();
    println!("Wrote plot_Fx_vs_x.svg");

    println!("\nOpen SVGs in browser to inspect.");
}
