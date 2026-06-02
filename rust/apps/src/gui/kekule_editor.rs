use std::collections::HashMap;
use surfmol_common::math::vec3::Vec3d;
use surfmol_topology::builder::{Builder, AtomH, BondH};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EditMode {
    Select,
    HexPaint,
    HexToggle,
    AtomDraw,
    BondDraw,
}

impl Default for EditMode {
    fn default() -> Self { EditMode::Select }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AtomType { C, N, O, H }

impl Default for AtomType {
    fn default() -> Self { AtomType::C }
}

impl AtomType {
    pub fn as_str(&self) -> &'static str {
        match self { AtomType::C => "C", AtomType::N => "N", AtomType::O => "O", AtomType::H => "H" }
    }
}

/// State and parameters for Kekule-style hex grid molecular editing.
#[derive(Default)]
pub struct KekuleEditor {
    pub edit_mode: EditMode,
    pub atom_type: AtomType,
    pub auto_h_cap: bool,
    pub auto_bonds: bool,
    pub grid_mode: bool,
    pub pick_radius: f64,
    pub a_cc: f64,
    // Ribbon generation parameters
    pub ribbon_rows: i32,
    pub ribbon_bottom: String,
    pub ribbon_top: String,
    pub ribbon2_rows: i32,
    pub ribbon2_bottom: String,
    pub ribbon2_top: String,
    pub ribbon_l_hb: f64,
}

impl KekuleEditor {
    pub fn new() -> Self {
        Self {
            edit_mode: EditMode::Select,
            atom_type: AtomType::C,
            auto_h_cap: true,
            auto_bonds: true,
            grid_mode: false,
            pick_radius: 0.5,
            a_cc: 1.42,
            ribbon_rows: 4,
            ribbon_bottom: "n".to_string(),
            ribbon_top: "n".to_string(),
            ribbon2_rows: 4,
            ribbon2_bottom: "n".to_string(),
            ribbon2_top: "n".to_string(),
            ribbon_l_hb: 3.0,
        }
    }

    /// Handle a click at world-space position (on the xy plane, z=0).
    /// Returns true if the builder was modified.
    pub fn on_click(&mut self, builder: &mut Builder, pos_ws: Vec3d) -> bool {
        match self.edit_mode {
            EditMode::HexPaint => {
                let (q, r) = Builder::snap_to_ring(pos_ws.x, pos_ws.y, self.a_cc);
                builder.add_hex_ring(q, r, self.a_cc);
                true
            }
            EditMode::HexToggle => {
                let (q, r) = Builder::snap_to_ring(pos_ws.x, pos_ws.y, self.a_cc);
                builder.toggle_hex_ring(q, r, self.a_cc);
                true
            }
            EditMode::AtomDraw => {
                let el = self.atom_type.as_str();
                // 1. Check if we're near an existing atom
                if let Some((ah, _)) = builder.find_nearest_atom(pos_ws, self.pick_radius) {
                    builder.set_atom_element(ah, el);
                    true
                } else if self.grid_mode {
                    // 2. Grid mode: snap to nearest grid node
                    let pin = Builder::snap_to_grid(pos_ws.x, pos_ws.y);
                    if let Some(&ah) = builder.pin_to_atom.get(&pin) {
                        if builder.is_atom_alive(ah) {
                            builder.set_atom_element(ah, el);
                            return true;
                        }
                    }
                    // No atom at this grid node: add new one at grid position
                    let snapped_pos = Vec3d::new(pin.0 as f64, pin.1 as f64, 0.0);
                    builder.add_atom(snapped_pos, el);
                    true
                } else {
                    // 3. Free mode: add atom at exact click position
                    builder.add_atom(pos_ws, el);
                    true
                }
            }
            _ => false,
        }
    }

    /// Handle right-click (context action).
    pub fn on_right_click(&mut self, builder: &mut Builder, pos_ws: Vec3d) -> bool {
        match self.edit_mode {
            EditMode::HexPaint | EditMode::HexToggle => {
                let (q, r) = Builder::snap_to_ring(pos_ws.x, pos_ws.y, self.a_cc);
                builder.remove_hex_ring(q, r, false, self.a_cc);
                true
            }
            EditMode::AtomDraw => {
                if let Some((ah, _)) = builder.find_nearest_atom(pos_ws, self.pick_radius) {
                    builder.remove_atom(ah);
                    true
                } else { false }
            }
            _ => false,
        }
    }

    pub fn set_edit_mode(&mut self, mode: EditMode) { self.edit_mode = mode; }
    pub fn set_atom_type(&mut self, atype: AtomType) { self.atom_type = atype; }
    pub fn toggle_auto_h_cap(&mut self) { self.auto_h_cap = !self.auto_h_cap; }
    pub fn toggle_auto_bonds(&mut self) { self.auto_bonds = !self.auto_bonds; }
    pub fn toggle_grid_mode(&mut self) { self.grid_mode = !self.grid_mode; }
    pub fn set_pick_radius(&mut self, r: f64) { self.pick_radius = r; }
}

/// Parse passivation string into group names.
/// Encoding: n->NH, N->N, o->C=O, O->O, H->CH, h->C-OH
pub fn parse_passivation_string(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for c in s.chars() {
        let group = match c {
            'n' => "NH", 'N' => "N", 'o' => "C=O", 'O' => "O",
            'H' => "CH", 'h' => "C-OH", _ => "H",
        };
        out.push(group.to_string());
    }
    out
}

/// Build a zigzag ribbon in the given Builder.
pub fn build_zigzag_ribbon(builder: &mut Builder, width_chains: i32, length_cells: i32, passivation_bottom: &[String], passivation_top: &[String], a_cc: f64, scale_x: f64) {
    // TODO: port from KekuleBackend.py
    // For now, generate a simple hex grid strip
    let s3 = 3.0_f64.sqrt();
    let dy = a_cc * 1.5;
    let dx = a_cc * s3;
    let mut nodes: HashMap<(i32, i32), AtomH> = HashMap::new();
    for iy in 0..width_chains {
        for ix in 0..length_cells {
            let x = (ix as f64) * dx * scale_x + if iy % 2 == 1 { dx * 0.5 * scale_x } else { 0.0 };
            let y = (iy as f64) * dy;
            let pin = Builder::snap_to_grid(x, y);
            if let Some(&ah) = builder.pin_to_atom.get(&pin) {
                if builder.is_atom_alive(ah) {
                    nodes.insert((ix, iy), ah);
                    continue;
                }
            }
            let pos = Vec3d::new(x, y, 0.0);
            let ah = builder.add_atom(pos, "C");
            builder.atom_mut(ah).pin = Some(pin);
            builder.atom_mut(ah).hybridization = 2;
            builder.pin_to_atom.insert(pin, ah);
            nodes.insert((ix, iy), ah);
        }
    }
    // connect neighbors at bond distance
    let bond_cutoff_sq = (a_cc * 1.1).powi(2);
    for (&(ix, iy), &ah) in &nodes {
        for (&(jx, jy), &bh) in &nodes {
            if ix >= jx && iy >= jy { continue; }
            let pa = builder.atom(ah).pos;
            let pb = builder.atom(bh).pos;
            let d2 = Vec3d::set_sub(pb, pa).norm2();
            if d2 < bond_cutoff_sq {
                builder.add_bond(ah, bh, 1);
            }
        }
    }
}

/// Combine two ribbons with H-bond spacing.
pub fn combine_ribbons(bottom: &mut Builder, top: &mut Builder, l_hb: f64, _shift_x: f64) {
    // TODO: proper merge; for now just add all atoms from top to bottom
    let top_atoms: Vec<(Vec3d, String)> = top.iter_atoms().map(|(_, ad)| (ad.pos, ad.element.clone())).collect();
    let mut new_handles = Vec::new();
    for (pos, el) in top_atoms {
        let mut shifted = pos;
        shifted.y += l_hb;
        let ah = bottom.add_atom(shifted, &el);
        new_handles.push(ah);
    }
    // TODO: copy bonds and other metadata
}

/// Adjust hydrogen caps for under-coordinated C atoms.
pub fn adjust_h_caps(builder: &mut Builder, bond_length: f64) {
    let mut to_add: Vec<(AtomH, Vec3d)> = Vec::new();
    for (ah, ad) in builder.iter_atoms() {
        if ad.element != "C" { continue; }
        let missing = 4 - ad.nbond as i32;
        if missing <= 0 { continue; }
        let pos = ad.pos;
        // simple: place H above in +y direction
        for i in 0..missing {
            let angle = (i as f64) * (std::f64::consts::PI * 2.0 / missing as f64);
            let h_pos = Vec3d::new(pos.x + bond_length * angle.cos(), pos.y + bond_length * angle.sin(), pos.z);
            to_add.push((ah, h_pos));
        }
    }
    // TODO: proper geometry based on existing neighbors
    for (_parent, h_pos) in to_add {
        let _ah = builder.add_atom(h_pos, "H");
        // TODO: set parent, is_h_cap
    }
}

/// Rebuild the builder from positions using radii.
pub fn rebuild_bonds(builder: &mut Builder) {
    // TODO: recompute bonds by distance cutoff
}

/// Export builder atoms to XYZ string.
pub fn export_xyz(builder: &Builder) -> String {
    let mut lines = Vec::new();
    let atoms: Vec<_> = builder.iter_atoms().collect();
    lines.push(format!("{}", atoms.len()));
    lines.push("Generated by SurfMol KekuleEditor".to_string());
    for (_, ad) in atoms {
        lines.push(format!("{:2} {:12.6} {:12.6} {:12.6}", ad.element, ad.pos.x, ad.pos.y, ad.pos.z));
    }
    lines.join("\n")
}

/// Return a string summary of the builder contents.
pub fn builder_summary(builder: &Builder) -> String {
    let mut n_c = 0; let mut n_n = 0; let mut n_o = 0; let mut n_h = 0; let mut n_other = 0;
    for (_, ad) in builder.iter_atoms() {
        match ad.element.as_str() {
            "C" => n_c += 1, "N" => n_n += 1, "O" => n_o += 1, "H" => n_h += 1, _ => n_other += 1,
        }
    }
    format!("C={} N={} O={} H={} other={}", n_c, n_n, n_o, n_h, n_other)
}

/// RGBA color for an element (for visualization).
pub fn element_color(el: &str) -> [f32; 4] {
    match el {
        "C" => [0.5f32, 0.5, 0.5, 1.0],
        "N" => [0.2f32, 0.2, 1.0, 1.0],
        "O" => [1.0f32, 0.2, 0.2, 1.0],
        "H" => [0.9f32, 0.9, 0.9, 1.0],
        _   => [0.7f32, 0.7, 0.7, 1.0],
    }
}

/// Collect hex grid reference points (2D lattice of dots) for empty neighbors of existing tiles.
pub fn collect_hex_grid_points(hex_tiles: &std::collections::HashSet<(i32, i32)>, a_cc: f64) -> Vec<Vec3d> {
    let mut points = std::collections::HashSet::new();
    let dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)];
    for &(q, r) in hex_tiles {
        for &(dq, dr) in &dirs {
            let nq = q + dq; let nr = r + dr;
            if !hex_tiles.contains(&(nq, nr)) {
                let nodes = Builder::honeycomb_ring_nodes(nq, nr, a_cc);
                for node in nodes {
                    let key = ((node.0 * 1e6).round() as i64, (node.1 * 1e6).round() as i64);
                    points.insert(key);
                }
            }
        }
    }
    points.into_iter().map(|(x, y)| Vec3d::new(x as f64 / 1e6, y as f64 / 1e6, 0.0)).collect()
}

/// Collect hex grid lines for visualization.
pub fn collect_hex_lines(hex_tiles: &std::collections::HashSet<(i32, i32)>, a_cc: f64) -> Vec<(Vec3d, Vec3d)> {
    let mut lines = Vec::new();
    for &(q, r) in hex_tiles {
        let nodes = Builder::honeycomb_ring_nodes(q, r, a_cc);
        for i in 0..6 {
            let a = Vec3d::new(nodes[i].0, nodes[i].1, 0.0);
            let b = Vec3d::new(nodes[(i + 1) % 6].0, nodes[(i + 1) % 6].1, 0.0);
            lines.push((a, b));
        }
    }
    lines
}

/// Collect builder bonds as line segments for visualization.
pub fn collect_builder_bonds(builder: &Builder) -> Vec<(Vec3d, Vec3d)> {
    let mut lines = Vec::new();
    for (_, bd) in builder.iter_bonds() {
        if !builder.is_atom_alive(bd.a) || !builder.is_atom_alive(bd.b) { continue; }
        let pa = builder.atom(bd.a).pos;
        let pb = builder.atom(bd.b).pos;
        lines.push((pa, pb));
    }
    lines
}

/// Collect builder atom positions for visualization.
pub fn collect_builder_atoms(builder: &Builder) -> Vec<(Vec3d, String)> {
    let mut out = Vec::new();
    for (_, ad) in builder.iter_atoms() {
        out.push((ad.pos, ad.element.clone()));
    }
    out
}

/// Collect ghost hex outlines for empty grid cells near existing tiles.
pub fn collect_ghost_hexes(hex_tiles: &std::collections::HashSet<(i32, i32)>, a_cc: f64) -> Vec<(Vec3d, Vec3d)> {
    let mut neighbors = std::collections::HashSet::new();
    let dirs = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)];
    for &(q, r) in hex_tiles {
        for &(dq, dr) in &dirs {
            let nq = q + dq; let nr = r + dr;
            if !hex_tiles.contains(&(nq, nr)) {
                neighbors.insert((nq, nr));
            }
        }
    }
    collect_hex_lines(&neighbors, a_cc)
}
