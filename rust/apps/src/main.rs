use macroquad::prelude::*;
use macroquad::models::{Mesh, Vertex, draw_mesh};
use surfmol_forcefields::mol_world::MolWorld;
use surfmol_forcefields::mol_world::BondedFFMode;
use surfmol_forcefields::nonbonded::NonBondedFF;
use surfmol_common::math::vec3::Vec3d;
use surfmol_common::xyz;
use surfmol_topology::builder;
use surfmol_topology::assign_uff;
use surfmol_topology::params::{Params, get_reqh};
use std::collections::HashMap;
use std::path::PathBuf;

// ------------------------------------------------------------------
// Constants
// ------------------------------------------------------------------
const ATOM_SCALE: f32 = 0.25;          // visual scale multiplier for atom radii
const BOND_THICKNESS: f32 = 0.05;     // visual thickness for bonds
const K_PICK: f64 = 30.0;             // spring constant for mouse-picking force [eV/A^2]
const PER_FRAME: i32 = 100;           // MD steps per frame redraw (like C++ perFrame)
const PICK_RAY_R: f32 = 0.5;          // picking radius used in C++ pickParticle(ray0,hray,R=0.5)
const SURFACE_GRID_N: i32 = 256;      // surface grid resolution (texture size)
const SURFACE_SIZE: f32 = 10.0;       // surface half-extent in Å
const SURFACE_Z0: f32 = 0.0;          // surface plane z
const LATTICE_A: f64 = 3.5;           // NaCl lattice constant
const BETA_VDW: f64 = 0.5;            // vdW decay
const Q_AMP: f64 = 1.0;               // electrostatic amplitude
const PLQ_AMP: f64 = 1.0;             // Pauli/London amplitude

const GROUP_SIZE_DEFAULT: usize = 32;

// ------------------------------------------------------------------
// Element color / radius lookup from loaded Params
// ------------------------------------------------------------------
fn element_color(elem: &str, params: &Params) -> Color {
    params.get_element_type(elem)
        .map(|et| u32_to_color(et.color))
        .unwrap_or_else(|| Color::from_rgba(200, 200, 200, 255))
}

fn element_radius(elem: &str, params: &Params) -> f32 {
    params.get_element_type(elem)
        .map(|et| et.r_vdw as f32)
        .unwrap_or(1.0)
}

fn u32_to_color(c: u32) -> Color {
    Color::from_rgba(
        ((c >> 16) & 0xFF) as u8,
        ((c >> 8)  & 0xFF) as u8,
        (c & 0xFF) as u8,
        255,
    )
}

#[derive(Copy, Clone, Debug)]
struct Aabb { min: Vec3, max: Vec3 }

fn draw_aabb(b: Aabb, col: Color) {
    let mn = b.min;
    let mx = b.max;
    let p000 = vec3(mn.x, mn.y, mn.z);
    let p001 = vec3(mn.x, mn.y, mx.z);
    let p010 = vec3(mn.x, mx.y, mn.z);
    let p011 = vec3(mn.x, mx.y, mx.z);
    let p100 = vec3(mx.x, mn.y, mn.z);
    let p101 = vec3(mx.x, mn.y, mx.z);
    let p110 = vec3(mx.x, mx.y, mn.z);
    let p111 = vec3(mx.x, mx.y, mx.z);
    draw_line_3d(p000, p001, col); draw_line_3d(p000, p010, col); draw_line_3d(p000, p100, col);
    draw_line_3d(p111, p110, col); draw_line_3d(p111, p101, col); draw_line_3d(p111, p011, col);
    draw_line_3d(p001, p011, col); draw_line_3d(p001, p101, col);
    draw_line_3d(p010, p011, col); draw_line_3d(p010, p110, col);
    draw_line_3d(p100, p101, col); draw_line_3d(p100, p110, col);
}

#[inline(always)]
fn color_group(g: usize) -> Color {
    let t = (g as f32 * 0.271828).fract();
    Color::new(0.2 + 0.7 * t, 0.2 + 0.7 * (1.0 - t), 0.5 + 0.4 * (t - 0.5).abs(), 1.0)
}

#[inline(always)]
fn morton3_10(x: u32, y: u32, z: u32) -> u32 {
    #[inline(always)]
    fn part1by2(mut v: u32) -> u32 {
        v &= 0x000003ff;
        v = (v | (v << 16)) & 0x30000ff;
        v = (v | (v << 8)) & 0x300f00f;
        v = (v | (v << 4)) & 0x30c30c3;
        v = (v | (v << 2)) & 0x9249249;
        v
    }
    (part1by2(x) << 0) | (part1by2(y) << 1) | (part1by2(z) << 2)
}

fn compute_groups_morton(pos: &[Vec3], group_size: usize) -> (Vec<usize>, Vec<Aabb>) {
    let n = pos.len();
    if n == 0 { return (Vec::new(), Vec::new()); }
    let mut mn = pos[0];
    let mut mx = pos[0];
    for &p in pos.iter() {
        mn.x = mn.x.min(p.x); mn.y = mn.y.min(p.y); mn.z = mn.z.min(p.z);
        mx.x = mx.x.max(p.x); mx.y = mx.y.max(p.y); mx.z = mx.z.max(p.z);
    }
    let span = (mx - mn).max(vec3(1e-6, 1e-6, 1e-6));
    let mut keys: Vec<(u32, usize)> = (0..n).map(|i| {
        let p = (pos[i] - mn) / span;
        let xi = (p.x.clamp(0.0, 1.0) * 1023.0) as u32;
        let yi = (p.y.clamp(0.0, 1.0) * 1023.0) as u32;
        let zi = (p.z.clamp(0.0, 1.0) * 1023.0) as u32;
        (morton3_10(xi, yi, zi), i)
    }).collect();
    keys.sort_unstable_by_key(|p| p.0);
    let order: Vec<usize> = keys.into_iter().map(|p| p.1).collect();
    let ng = (n + group_size - 1) / group_size;
    let mut bbs = vec![Aabb { min: vec3(1e10, 1e10, 1e10), max: vec3(-1e10, -1e10, -1e10) }; ng];
    for g in 0..ng {
        let i0 = g * group_size;
        let i1 = ((g + 1) * group_size).min(n);
        let mut b = bbs[g];
        for ii in i0..i1 {
            let i = order[ii];
            let p = pos[i];
            b.min.x = b.min.x.min(p.x); b.min.y = b.min.y.min(p.y); b.min.z = b.min.z.min(p.z);
            b.max.x = b.max.x.max(p.x); b.max.y = b.max.y.max(p.y); b.max.z = b.max.z.max(p.z);
        }
        bbs[g] = b;
    }
    (order, bbs)
}

// ------------------------------------------------------------------
// Vec3d  <->  macroquad Vec3
// ------------------------------------------------------------------
#[inline(always)]
fn to_mq(v: Vec3d) -> Vec3 { vec3(v.x as f32, v.y as f32, v.z as f32) }

#[inline(always)]
fn to_v3d(v: Vec3) -> Vec3d { Vec3d::new(v.x as f64, v.y as f64, v.z as f64) }

// ------------------------------------------------------------------
// Spring force pulling atom toward mouse ray (perpendicular)
// Matches FireCore getForceSpringRay
// ------------------------------------------------------------------
fn get_force_spring_ray(p: Vec3, hray: Vec3, ray0: Vec3, k: f32) -> Vec3 {
    let dp = p - ray0;
    let cdot = hray.dot(dp);
    let dp_perp = dp - hray * cdot;
    -dp_perp * k  // pull toward ray
}

// ------------------------------------------------------------------
// 3D ray ↔ sphere intersection  (returns t along ray if hit)
// ------------------------------------------------------------------
fn ray_sphere(ro: Vec3, rd: Vec3, sc: Vec3, sr: f32) -> Option<f32> {
    let oc = ro - sc;
    let b = oc.dot(rd);
    let c = oc.dot(oc) - sr * sr;
    let disc = b * b - c;
    if disc < 0.0 { return None; }
    let t = -b - disc.sqrt();
    if t >= 0.0 { Some(t) } else { None }
}

// ------------------------------------------------------------------
// Camera controller (orbit-style)
// ------------------------------------------------------------------
// Trackball camera (quaternion-based, proper arcball rotation)
// ------------------------------------------------------------------
struct TrackballCam {
    target: Vec3,
    rotation: Quat,  // orientation: applies to camera-local axes
    dist_cam: f32,  // camera distance from target (does NOT control ortho zoom)
    zoom: f32,      // C++-style zoom = half-height of ortho view volume in world units
}

impl TrackballCam {
    fn new(target: Vec3, dist: f32) -> Self {
        // Start with camera at +Z looking toward target
        Self { target, rotation: Quat::IDENTITY, dist_cam: dist, zoom: dist }
    }

    fn pos(&self) -> Vec3 {
        // Camera offset in world space: rotate (0,0,dist) by quaternion
        self.target + self.rotation * vec3(0.0, 0.0, self.dist_cam)
    }

    fn fwd(&self) -> Vec3 {
        // Camera looks toward target: -local Z
        (self.target - self.pos()).normalize()
    }

    fn up(&self) -> Vec3 {
        self.rotation * vec3(0.0, 1.0, 0.0)
    }

    fn right(&self) -> Vec3 {
        self.fwd().cross(self.up()).normalize()
    }

    fn to_mq_camera(&self) -> Camera3D {
        Camera3D {
            position: self.pos(),
            target:   self.target,
            up:       self.up(),
            fovy:     self.zoom * 2.0, // macroquad ortho: fovy behaves as view height; width follows from aspect
            projection: Projection::Orthographics,
            ..Default::default()
        }
    }

    fn zoom(&mut self, delta: f32) {
        self.zoom *= 1.0 + delta * 0.1;
        self.zoom = self.zoom.clamp(0.2, 200.0);
    }

    /// Pan target in view plane.
    fn pan(&mut self, dx: f32, dy: f32) {
        let r = self.right();
        let u = self.up();
        // Speed proportional to zoom
        let s = self.zoom * 0.001;
        self.target += r * (-dx * s) + u * (dy * s);
    }

    /// Map mouse screen position to a point on the unit trackball sphere.
    fn mouse_to_sphere(&self, mouse: Vec2) -> Vec3 {
        let radius = screen_width().min(screen_height()) * 0.4;
        let x = (mouse.x - screen_width() * 0.5) / radius;
        let y = -(mouse.y - screen_height() * 0.5) / radius;
        let r2 = x * x + y * y;
        if r2 <= 1.0 {
            vec3(x, y, (1.0 - r2).sqrt())
        } else {
            let s = 1.0 / r2.sqrt();
            vec3(x * s, y * s, 0.0)
        }
    }

    /// Apply a trackball rotation from previous mouse to current mouse.
    fn rotate(&mut self, prev: Vec2, curr: Vec2) {
        let a = self.mouse_to_sphere(prev);
        let b = self.mouse_to_sphere(curr);
        let q = Quat::from_rotation_arc(a, b);
        self.rotation = (q * self.rotation).normalize();
    }

    /// Build a world-space ray from mouse position.
    /// In ortho: origin on near plane through camera, direction = view dir.
    fn screen_ray(&self, mouse: Vec2) -> (Vec3, Vec3) {
        let fwd = self.fwd();
        let right = self.right();
        let up = self.up();

        // Match C++ AppSDL2OGL_3D::mouseHandling():
        // mouse_begin_x = (2*mouseX - WIDTH )*zoom/HEIGHT;
        // mouse_begin_y = (2*mouseY - HEIGHT)*zoom/HEIGHT;   with mouseY flipped (HEIGHT-mouseY)
        let w = screen_width();
        let h = screen_height();
        let mx = mouse.x;
        let my = h - mouse.y; // flip Y like SDL code
        let mouse_begin_x = (2.0 * mx - w) * self.zoom / h;
        let mouse_begin_y = (2.0 * my - h) * self.zoom / h;

        let origin = self.pos() + right * mouse_begin_x + up * mouse_begin_y;
        (origin, fwd)
    }
}

// ------------------------------------------------------------------
// Surface potential sample point
// ------------------------------------------------------------------
struct SurfSample {
    pos: Vec3,
    pot: f32,   // electrostatic potential (unit charge)
}

// ------------------------------------------------------------------
// App state
// ------------------------------------------------------------------
struct App {
    world: MolWorld,
    elems: Vec<String>,
    params: Params,
    cam: TrackballCam,

    // interaction
    selected: Option<usize>,
    pinned: Vec<bool>,
    trackballing: bool,
    trackball_prev: Vec2,
    pick_k: f64,
    per_frame: i32,

    // MolGUI-like mouse gesture state (click vs drag)
    lmb_down: bool,
    mouse_down: Vec2,
    ray0_down: Vec3,

    // display toggles
    show_bonds: bool,
    show_surface: bool,
    show_help: bool,
    show_groups: bool,
    show_ports: bool,

    group_size: usize,
    group_order: Vec<usize>,
    group_bboxes: Vec<Aabb>,

    // physics
    run_relax: bool,
    dt: f64,
    flim: f64,
    damping: f64,
    cdamp: f64,
    f2conv: f64,

    // cached geometry
    apos: Vec<Vec3>,      // atom positions in mq Vec3
    bonds: Vec<[usize; 2]>, // bond atom indices
    surf_samples: Vec<SurfSample>,
    surface_texture: Option<Texture2D>,
    etot: f64,
    eb: f64, ea: f64, ed: f64, ei: f64, enb: f64, es: f64,

    // mouse tracking
    prev_mouse: Vec2,
}

impl App {
    fn new(world: MolWorld, elems: Vec<String>, params: Params, group_size: usize) -> Self {
        let natoms = world.natoms();
        let mut app = Self {
            world,
            elems,
            params,
            cam: TrackballCam::new(vec3(0.0, 2.0, 0.0), 6.0),
            selected: None,
            pinned: vec![false; natoms],
            trackballing: false,
            trackball_prev: Vec2::ZERO,
            pick_k: K_PICK,
            per_frame: PER_FRAME,
            lmb_down: false,
            mouse_down: Vec2::ZERO,
            ray0_down: Vec3::ZERO,
            show_bonds: true,
            show_surface: true,
            show_help: true,
            show_groups: false,
            show_ports: false,
            group_size,
            group_order: Vec::new(),
            group_bboxes: Vec::new(),
            run_relax: false,
            dt: 0.02,
            flim: 1000.0,
            damping: 0.05,
            cdamp: 0.95,
            f2conv: 1e-6,
            apos: vec![Vec3::ZERO; natoms],
            bonds: vec![],
            surf_samples: vec![],
            surface_texture: None,
            etot: 0.0, eb: 0.0, ea: 0.0, ed: 0.0, ei: 0.0, enb: 0.0, es: 0.0,
            prev_mouse: Vec2::ZERO,
        };
        app.rebuild_bond_cache();
        app.rebuild_surface_cache();
        app.sync_pos_from_engine();
        app.rebuild_groups();
        app.eval_energies();
        app
    }

    fn rebuild_groups(&mut self) {
        let (ord, mut bbs) = compute_groups_morton(&self.apos, self.group_size);
        for b in bbs.iter_mut() {
            b.min -= vec3(0.5, 0.5, 0.5);
            b.max += vec3(0.5, 0.5, 0.5);
        }
        self.group_order = ord;
        self.group_bboxes = bbs;
    }

    fn rebuild_bond_cache(&mut self) {
        self.bonds.clear();
        let uff = &self.world.uff;
        for ib in 0..uff.nbonds as usize {
            let b = uff.bon_atoms.as_slice()[ib];
            self.bonds.push([b[0] as usize, b[1] as usize]);
        }
    }

    /// Sample the NaCl surface electrostatic potential on a grid and bake into a texture.
    fn rebuild_surface_cache(&mut self) {
        self.surf_samples.clear();
        let Some(ref surf) = self.world.surface else { return };
        let z0 = SURFACE_Z0;
        let n = SURFACE_GRID_N;
        let size = SURFACE_SIZE;
        let dummy_req = [0.0, 0.0, 1.0, 0.0]; // unit charge, no Pauli/London
        let dummy_plq = surfmol_forcefields::surface::SurfaceFolded::req2plq(dummy_req, 2.0);
        let w = (n + 1) as usize;
        let h = (n + 1) as usize;
        let mut pixels = vec![0u8; w * h * 4];

        for iy in 0..=n {
            let y = -size + (2.0 * size) * (iy as f32 / n as f32);
            for ix in 0..=n {
                let x = -size + (2.0 * size) * (ix as f32 / n as f32);
                let pos = Vec3d::new(x as f64, y as f64, z0 as f64);
                let (e, _) = surf.eval_atom(pos, dummy_plq);
                let pot = e as f32;
                self.surf_samples.push(SurfSample { pos: vec3(x, y, z0), pot });
                let color = potential_color(pot);
                let rgba: [u8; 4] = color.into();
                let row = (n - iy) as usize; // flip Y for texture orientation
                let pix = (row * w + ix as usize) * 4;
                pixels[pix..pix + 4].copy_from_slice(&rgba);
            }
        }
        self.surface_texture = Some(Texture2D::from_rgba8(w as u16, h as u16, &pixels));
    }

    fn sync_pos_from_engine(&mut self) {
        let slice = self.world.dyn_atoms.atoms.apos.as_slice();
        for i in 0..self.world.natoms() {
            self.apos[i] = to_mq(slice[i]);
        }
    }

    fn sync_pos_to_engine(&mut self) {
        let n = self.world.natoms();
        let slice = self.world.dyn_atoms.atoms.apos.as_mut_slice();
        for i in 0..n {
            slice[i] = to_v3d(self.apos[i]);
        }
    }

    fn eval_energies(&mut self) {
        let (eb, ea, ed, ei, enb, es) = self.world.eval_forces();
        self.eb = eb; self.ea = ea; self.ed = ed; self.ei = ei; self.enb = enb; self.es = es;
        self.etot = eb + ea + ed + ei + enb + es;
    }

    fn pick_atom(&self, mouse: Vec2) -> Option<usize> {
        let (ro, rd) = self.cam.screen_ray(mouse);
        let mut best_t = f32::MAX;
        let mut best_i = None;
        for i in 0..self.world.natoms() {
            // Exactly like C++ pickParticle(): pick the sphere hit with minimum intersection t
            if let Some(t) = ray_sphere(ro, rd, self.apos[i], PICK_RAY_R) {
                if t < best_t {
                    best_t = t;
                    best_i = Some(i);
                }
            }
        }
        best_i
    }

    fn project_mouse_to_plane(&self, mouse: Vec2, plane_point: Vec3, plane_normal: Vec3) -> Vec3 {
        let (ro, rd) = self.cam.screen_ray(mouse);
        let denom = rd.dot(plane_normal);
        if denom.abs() < 1e-6 { return plane_point; }
        let t = (plane_point - ro).dot(plane_normal) / denom;
        ro + rd * t
    }

    fn do_relax_step(&mut self) {
        // If simulation is stopped, nothing moves. Picking must not implicitly run MD.
        if !self.run_relax { return; }

        for _ in 0..self.per_frame {
            let (eb, ea, ed, ei, enb, es) = self.world.eval_forces();
            self.eb = eb; self.ea = ea; self.ed = ed; self.ei = ei; self.enb = enb; self.es = es;
            self.etot = eb + ea + ed + ei + enb + es;

            // Apply spring force to picked atom (MolGUI-style)
            if let Some(idx) = self.selected {
                let atom_pos = to_mq(self.world.dyn_atoms.atoms.apos.as_slice()[idx]);
                let (ray0, hray) = self.cam.screen_ray(self.prev_mouse);
                let f_spring = get_force_spring_ray(atom_pos, hray, ray0, self.pick_k as f32);
                let fapos = self.world.dyn_atoms.fapos.as_mut_slice();
                fapos[idx].x += f_spring.x as f64;
                fapos[idx].y += f_spring.y as f64;
                fapos[idx].z += f_spring.z as f64;
            }

            let mut vf = 0.0;
            for ia in 0..self.world.natoms() {
                if self.pinned[ia] { continue; }
                let (_, _, vf_i) = self.world.move_atom_md(ia, self.dt, self.flim, self.cdamp);
                vf += vf_i;
            }
            self.sync_pos_from_engine();
        }
    }

    // ------------------------------------------------------------------
    // Input handling
    // ------------------------------------------------------------------
    fn handle_input(&mut self) {
        let mouse = mouse_position();
        let mouse_now = vec2(mouse.0, mouse.1);
        let mouse_delta = mouse_now - self.prev_mouse;
        self.prev_mouse = mouse_now;

        // Toggle help
        if is_key_pressed(KeyCode::H) {
            self.show_help = !self.show_help;
        }
        if is_key_pressed(KeyCode::T) {
            self.show_ports = !self.show_ports;
        }
        if is_key_pressed(KeyCode::F) {
            self.world.bonded_mode = match self.world.bonded_mode {
                BondedFFMode::Uff => BondedFFMode::RigidSp3,
                BondedFFMode::RigidSp3 => BondedFFMode::Uff,
            };
            println!("bonded_mode = {:?}", self.world.bonded_mode);
        }
        // Toggle surface
        if is_key_pressed(KeyCode::S) {
            self.show_surface = !self.show_surface;
        }
        if is_key_pressed(KeyCode::G) {
            self.show_groups = !self.show_groups;
        }
        // Toggle bonds
        if is_key_pressed(KeyCode::B) {
            self.show_bonds = !self.show_bonds;
        }
        // Toggle relaxation (Space)
        if is_key_pressed(KeyCode::Space) {
            self.run_relax = !self.run_relax;
        }

        // Picking stiffness tuning (like tweaking Kpick)
        if is_key_pressed(KeyCode::LeftBracket) {
            self.pick_k *= 0.8;
            println!("pick_k = {}", self.pick_k);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            self.pick_k *= 1.25;
            println!("pick_k = {}", self.pick_k);
        }

        // Per-frame integration tuning
        if is_key_pressed(KeyCode::Minus) {
            self.per_frame = (self.per_frame / 2).max(1);
            println!("per_frame = {}", self.per_frame);
        }
        if is_key_pressed(KeyCode::Equal) {
            self.per_frame = (self.per_frame * 2).min(2000);
            println!("per_frame = {}", self.per_frame);
        }
        // Pin/unpin selected atom (P)
        if is_key_pressed(KeyCode::P) {
            if let Some(idx) = self.selected {
                self.pinned[idx] = !self.pinned[idx];
            }
        }
        // Deselect (Escape)
        if is_key_pressed(KeyCode::Escape) {
            self.selected = None;
        }
        // Reset camera (C)
        if is_key_pressed(KeyCode::C) {
            self.cam = TrackballCam::new(vec3(0.0, 1.0, 0.0), 6.0);
        }

        // Mouse wheel = zoom
        let (_, wheel_y) = mouse_wheel();
        if wheel_y != 0.0 {
            self.cam.zoom(wheel_y);
        }

        // --- Left mouse button: MolGUI gesture (mouseStartSelectionBox on DOWN, pick on UP if small move) ---
        if is_mouse_button_pressed(MouseButton::Left) {
            self.lmb_down = true;
            self.mouse_down = mouse_now;
            self.ray0_down = self.cam.screen_ray(mouse_now).0;
        }
        if is_mouse_button_down(MouseButton::Left) {
            if is_key_down(KeyCode::LeftShift) || is_key_down(KeyCode::RightShift) {
                self.cam.pan(mouse_delta.x, mouse_delta.y);
            }
        }
        if is_mouse_button_released(MouseButton::Left) {
            self.lmb_down = false;
            // MolGUI uses ray0.dist2(ray0_start)<0.1 as click threshold; we do it in pixels.
            let dpix = (mouse_now - self.mouse_down).length();
            if dpix < 5.0 {
                // click -> pick/unpick
                if let Some(idx) = self.pick_atom(mouse_now) {
                    self.selected = if self.selected == Some(idx) { None } else { Some(idx) };
                } else {
                    self.selected = None;
                }
            } else {
                // drag gesture -> do not change selection (MolGUI would do selectRect here)
            }
        }

        // --- Right mouse button: trackball rotate ---
        if is_mouse_button_pressed(MouseButton::Right) {
            self.trackballing = true;
            self.trackball_prev = mouse_now;
        }
        if is_mouse_button_released(MouseButton::Right) {
            self.trackballing = false;
            self.selected = None; // matches MolGUI: RMB up clears ipicked
        }
        if self.trackballing && is_mouse_button_down(MouseButton::Right) {
            self.cam.rotate(self.trackball_prev, mouse_now);
            self.trackball_prev = mouse_now;
        }
    }

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------
    fn draw(&self) {
        clear_background(Color::from_rgba(20, 20, 30, 255));

        set_camera(&self.cam.to_mq_camera());

        let (ray0, hray) = self.cam.screen_ray(self.prev_mouse);
        let csz = 0.15;
        draw_line_3d(ray0 - vec3(csz, 0.0, 0.0), ray0 + vec3(csz, 0.0, 0.0), Color::from_rgba(255, 80, 80, 255));
        draw_line_3d(ray0 - vec3(0.0, csz, 0.0), ray0 + vec3(0.0, csz, 0.0), Color::from_rgba(255, 80, 80, 255));
        draw_line_3d(ray0 - vec3(0.0, 0.0, csz), ray0 + vec3(0.0, 0.0, csz), Color::from_rgba(255, 80, 80, 255));
        draw_line_3d(ray0, ray0 + hray * 2.0, Color::from_rgba(255, 80, 80, 255));

        // --- Surface potential grid ---
        if self.show_surface {
            self.draw_surface();
        }

        // --- Bonds ---
        if self.show_bonds {
            for b in &self.bonds {
                let p0 = self.apos[b[0]];
                let p1 = self.apos[b[1]];
                let mid = (p0 + p1) * 0.5;
                let dir = (p1 - p0).normalize();
                let len = (p1 - p0).length();
                // Draw as a thin cylinder (macroquad draw_cylinder is Y-up, centered)
                // We need to rotate it. Since macroquad doesn't support rotated cylinders easily,
                // draw multiple line segments for a smooth bond appearance.
                const SEG: i32 = 8;
                let col = Color::from_rgba(180, 180, 180, 255);
                for i in 0..SEG {
                    let t0 = i as f32 / SEG as f32;
                    let t1 = (i + 1) as f32 / SEG as f32;
                    let a = p0 + (p1 - p0) * t0;
                    let b_ = p0 + (p1 - p0) * t1;
                    draw_line_3d(a, b_, col);
                }
            }
        }

        // --- Atoms ---
        for i in 0..self.world.natoms() {
            let pos = self.apos[i];
            let r = if self.show_ports { 0.03 } else { element_radius(&self.elems[i], &self.params) * ATOM_SCALE };
            let col = element_color(&self.elems[i], &self.params);

            // Highlight selected atom with a bright, larger shell
            if self.selected == Some(i) {
                let (sel_col, ring_col) = if self.pinned[i] {
                    (Color::from_rgba(255, 200, 0, 200), YELLOW) // orange-yellow for pinned
                } else {
                    (Color::from_rgba(0, 255, 100, 200), GREEN)   // bright green for selected
                };
                draw_sphere(pos, r * 1.5, None, sel_col);
                // Wireframe ring around selected atom
                const N: i32 = 16;
                for k in 0..N {
                    let t0 = (k as f32 / N as f32) * std::f32::consts::TAU;
                    let t1 = ((k + 1) as f32 / N as f32) * std::f32::consts::TAU;
                    let p0r = pos + vec3(t0.cos(), t0.sin(), 0.0) * r * 1.6;
                    let p1r = pos + vec3(t1.cos(), t1.sin(), 0.0) * r * 1.6;
                    draw_line_3d(p0r, p1r, ring_col);
                }
            }

            draw_sphere(pos, r, None, col);
        }

        if self.show_ports {
            let uff = &self.world.uff;
            let rr = &self.world.rigid_sp3;
            let apos = self.world.dyn_atoms.atoms.apos.as_slice();
            let neigh_bs = self.world.dyn_atoms.atoms.neigh_bs.as_slice();
            for i in 0..self.world.natoms() {
                let xi = apos[i];
                let pi = vec3(xi.x as f32, xi.y as f32, xi.z as f32);
                let bs = neigh_bs[i].as_array();
                let np = rr.nport[i] as usize;
                for s in 0..np {
                    let ib = bs[s];
                    if ib < 0 { continue; }
                    let l0 = uff.bon_params.as_slice()[ib as usize][1];
                    let tip = rr.get_port_tip(apos, i, s, l0);
                    let pt = vec3(tip.x as f32, tip.y as f32, tip.z as f32);
                    draw_line_3d(pi, pt, ORANGE);
                    draw_sphere(pt, 0.05, None, Color::new(1.0, 0.4, 0.0, 0.9));
                }
            }
        }

        if self.show_groups {
            for (g, b) in self.group_bboxes.iter().enumerate() {
                draw_aabb(*b, color_group(g));
            }
        }

        // --- Draw line from picked atom to mouse cursor (like MolGUI) ---
        if let Some(idx) = self.selected {
            let atom_pos = self.apos[idx];
            draw_line_3d(atom_pos, ray0, RED);
            draw_sphere(ray0, 0.08, None, RED); // mouse cursor dot
        }

        // --- Axes helper (small, at origin) ---
        let o = vec3(0.0, 0.0, 0.0);
        draw_line_3d(o, vec3(1.0, 0.0, 0.0), RED);
        draw_line_3d(o, vec3(0.0, 1.0, 0.0), GREEN);
        draw_line_3d(o, vec3(0.0, 0.0, 1.0), BLUE);

        set_default_camera();

        // --- 2D UI overlay ---
        self.draw_ui();
    }

    fn draw_surface(&self) {
        let Some(ref texture) = self.surface_texture else { return };
        let size = SURFACE_SIZE;
        let z0 = SURFACE_Z0;
        let white: [u8; 4] = WHITE.into();
        let mesh = Mesh {
            vertices: vec![
                Vertex { position: vec3(-size, -size, z0), uv: vec2(0.0, 0.0), color: white, normal: vec4(0.0, 0.0, 1.0, 0.0) },
                Vertex { position: vec3( size, -size, z0), uv: vec2(1.0, 0.0), color: white, normal: vec4(0.0, 0.0, 1.0, 0.0) },
                Vertex { position: vec3(-size,  size, z0), uv: vec2(0.0, 1.0), color: white, normal: vec4(0.0, 0.0, 1.0, 0.0) },
                Vertex { position: vec3( size,  size, z0), uv: vec2(1.0, 1.0), color: white, normal: vec4(0.0, 0.0, 1.0, 0.0) },
            ],
            indices: vec![0, 1, 2, 1, 3, 2],
            texture: Some(texture.clone()),
        };
        draw_mesh(&mesh);
    }

    fn draw_ui(&self) {
        // Title
        draw_text("Molecule-on-Surface Viewer", 10.0, 24.0, 24.0, WHITE);

        // Energy info
        let y0 = 50.0;
        let dy = 18.0;
        draw_text(&format!("Etotal = {:10.4} eV", self.etot), 10.0, y0, 18.0, WHITE);
        draw_text(&format!("  bond={:8.3} angle={:8.3} dihed={:8.3}", self.eb, self.ea, self.ed), 10.0, y0 + dy, 16.0, GRAY);
        draw_text(&format!("  inv ={:8.3} nb  ={:8.3} surf={:8.3}", self.ei, self.enb, self.es), 10.0, y0 + dy * 2.0, 16.0, GRAY);

        // Selected atom info
        if let Some(idx) = self.selected {
            let sx = screen_width() - 300.0;
            draw_text(&format!("Atom {}: {}", idx, self.elems[idx]), sx, 30.0, 20.0, YELLOW);
            let pin_text = if self.pinned[idx] { "[PINNED]  Press P to unpin" } else { "Press P to pin" };
            draw_text(pin_text, sx, 52.0, 16.0, if self.pinned[idx] { ORANGE } else { GRAY });
            let pos = self.apos[idx];
            draw_text(&format!("pos: {:.3} {:.3} {:.3}", pos.x, pos.y, pos.z), sx, 72.0, 16.0, GRAY);
            let r = element_radius(&self.elems[idx], &self.params);
            draw_text(&format!("RvdW = {:.3} Å", r), sx, 90.0, 14.0, GRAY);
        }

        // Status / controls
        let relax_str = if self.run_relax { "ON  (press SPACE to pause)" } else { "OFF (press SPACE to run)" };
        draw_text(&format!("Relaxation: {}", relax_str), 10.0, screen_height() - 80.0, 18.0,
            if self.run_relax { GREEN } else { GRAY });

        // Help
        if self.show_help {
            let hx = 10.0;
            let hy = screen_height() - 180.0;
            draw_text("Controls:", hx, hy, 18.0, WHITE);
            let help = [
                "LMB click atom     -> pick/unpick (spring follow)",
                "Shift+LMB drag     -> pan camera",
                "RMB drag           -> rotate camera (trackball)",
                "RMB click          -> unpick atom",
                "Scroll             -> zoom",
                "SPACE              -> start/stop relaxation",
                "P                  -> pin/unpin picked atom",
                "S                  -> toggle surface",
                "B                  -> toggle bonds",
                "H                  -> toggle help",
                "ESC                -> unpick",
                "C                  -> reset camera",
                "G                  -> toggle group AABBs",
                "T                  -> toggle port visualization",
            ];
            for (i, line) in help.iter().enumerate() {
                draw_text(line, hx + 10.0, hy + 20.0 + i as f32 * 16.0, 14.0, GRAY);
            }
        } else {
            draw_text("Press H for help", 10.0, screen_height() - 40.0, 16.0, DARKGRAY);
        }
    }
}

// ------------------------------------------------------------------
// Color mapping for surface potential
// ------------------------------------------------------------------
fn potential_color(pot: f32) -> Color {
    let vmax = 1.0; // clip intensity (electrostatic map)
    let t = (pot / vmax).clamp(-1.0, 1.0);
    if t < 0.0 {
        // blue -> white smooth diverging
        let s = 1.0 + t; // s in [0,1]
        Color::from_rgba(
            (255.0 * s) as u8,
            (255.0 * s) as u8,
            255,
            255
        )
    } else {
        // white -> red smooth diverging
        let s = 1.0 - t; // s in [0,1]
        Color::from_rgba(
            255,
            (255.0 * s) as u8,
            (255.0 * s) as u8,
            255
        )
    }
}

// ------------------------------------------------------------------
// Window configuration
// ------------------------------------------------------------------
fn window_conf() -> macroquad::miniquad::conf::Conf {
    macroquad::miniquad::conf::Conf {
        window_title: "Molecule Surface Viewer".to_string(),
        window_width: 1200,
        window_height: 800,
        high_dpi: false,
        ..Default::default()
    }
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------
#[macroquad::main(window_conf)]
async fn main() {
    println!("Starting Molecule Surface Viewer...");

    // Determine workspace root (repo root) by finding Cargo.toml upward
    let workspace_root = std::env::current_dir()
        .and_then(|cwd| {
            let mut p = cwd.clone();
            loop {
                if p.join("Cargo.toml").exists() { break Ok(p); }
                if !p.pop() { break Ok(cwd.clone()); }
            }
        })
        .unwrap_or_else(|_| std::env::current_dir().unwrap());

    let args: Vec<String> = std::env::args().collect();
    let mut copies_x: usize = 1;
    let mut copies_y: usize = 1;
    let mut spacing: f64 = 12.0;
    let mut group_size: usize = GROUP_SIZE_DEFAULT;
    {
        let mut it = args.iter().skip(1);
        while let Some(a) = it.next() {
            match a.as_str() {
                "--copies-x" => { copies_x = it.next().unwrap_or(&"1".to_string()).parse().unwrap_or(1); }
                "--copies-y" => { copies_y = it.next().unwrap_or(&"1".to_string()).parse().unwrap_or(1); }
                "--spacing" => { spacing = it.next().unwrap_or(&"12.0".to_string()).parse().unwrap_or(12.0); }
                "--group-size" => { group_size = it.next().unwrap_or(&"32".to_string()).parse().unwrap_or(GROUP_SIZE_DEFAULT); }
                _ => {}
            }
        }
    }
    let xyz_path: PathBuf = args.iter().skip(1).find(|s| !s.starts_with("--")).map(|s| {
        let p = PathBuf::from(s);
        if p.is_absolute() { p } else { workspace_root.join(p) }
    }).unwrap_or_else(|| workspace_root.join("../data/xyz/pentacene.xyz"));

    println!("Loading XYZ: {:?}", xyz_path);
    let sys = xyz::read_xyz(&xyz_path).expect("read_xyz failed");
    println!("Loaded {} atoms", sys.elems.len());

    let mut apos = Vec::<Vec3d>::new();
    let mut elems = Vec::<String>::new();
    let mut charges = Vec::<f64>::new();
    for iy in 0..copies_y {
        for ix in 0..copies_x {
            let shift = Vec3d::new((ix as f64) * spacing, (iy as f64) * spacing, 0.0);
            let i0 = apos.len();
            apos.extend(sys.apos.iter().map(|p| Vec3d::set_add(*p, shift)));
            elems.extend(sys.elems.iter().cloned());
            charges.extend(sys.charges.iter().copied());
            let i1 = apos.len();
            assert!(i1 - i0 == sys.apos.len());
        }
    }
    println!("Spawned copies: {}x{} -> natoms={}", copies_x, copies_y, apos.len());

    let dat_dir_candidates = [workspace_root.join("tmp/FireCore_cpp/common_resources"), workspace_root.join("data")];
    let dat_dir = dat_dir_candidates.iter().find(|d| d.join("ElementTypes.dat").exists() && d.join("AtomTypes.dat").exists() && d.join("BondTypes.dat").exists() && d.join("AngleTypes.dat").exists()).cloned().unwrap_or_else(|| dat_dir_candidates[0].clone());
    let mut params = Params::new();
    let have_params = dat_dir.join("ElementTypes.dat").exists() && dat_dir.join("AtomTypes.dat").exists() && dat_dir.join("BondTypes.dat").exists() && dat_dir.join("AngleTypes.dat").exists();
    if have_params {
        params.load_element_types(dat_dir.join("ElementTypes.dat"));
        params.load_atom_types(dat_dir.join("AtomTypes.dat"));
        params.load_bond_types(dat_dir.join("BondTypes.dat"));
        params.load_angle_types(dat_dir.join("AngleTypes.dat"));
        if dat_dir.join("DihedralTypes.dat").exists() { params.load_dihedral_types(dat_dir.join("DihedralTypes.dat")); }
        println!("Loaded {} elements, {} atom types, {} bond types",
            params.elements.len(), params.atom_types.len(), params.bonds.len());
    } else {
        println!("WARNING: .dat files not found in {:?}; running with dummy radii/REQs/bond params", dat_dir);
    }

    let radii: Vec<f64> = if have_params {
        elems.iter().map(|el| {
            params.get_element_type(el).map(|et| et.r_cov).unwrap_or(1.0)
        }).collect()
    } else {
        elems.iter().map(|el| match el.as_str() { "H" => 0.31, "C" => 0.76, "N" => 0.71, "O" => 0.66, _ => 1.0 }).collect()
    };
    let mut b = builder::Builder::from_positions_and_radii(&apos, &radii, 0.4);
    let top = b.bake();
    let mut world = MolWorld::from_topology(&top);

    world.make_neigh_bs();
    world.bake_angle_neighs();
    world.bake_dihedral_neighs();
    world.bake_inversion_neighs();
    world.map_atom_interactions();

    world.nonbonded = Some(NonBondedFF::new(world.natoms()));
    let natoms = world.natoms();
    let neighs = world.dyn_atoms.atoms.neighs.as_slice().to_vec();
    world.nonbonded.as_mut().unwrap().make_second_neighs(&neighs, natoms);
    world.nonbonded.as_mut().unwrap().set_cutoff(8.0);

    {
        let neighs: Vec<[i32; 4]> = world.dyn_atoms.atoms.neighs.as_slice().iter().map(|q| q.as_array()).collect();
        let uff_types = assign_uff::assign_uff_types(&elems, &neighs);

        let mut counts: HashMap<String, usize> = HashMap::new();
        for t in &uff_types { *counts.entry(t.clone()).or_insert(0) += 1; }
        let mut kv: Vec<(String, usize)> = counts.into_iter().collect();
        kv.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        println!("=== UFF type histogram ===");
        for (t, c) in kv.iter() { println!("{:6}  {}", t, c); }

        let has_sp2 = uff_types.iter().any(|t| matches!(t.as_str(), "C_R"|"C_2"|"N_R"|"O_2"|"O_R"));
        if has_sp2 {
            world.bonded_mode = BondedFFMode::Uff;
            println!("Detected sp2/aromatic types -> default bonded_mode = Uff");
        }

        world.rigid_sp3.set_port_geometry_from_types(&uff_types);

        if have_params {

            for i in 0..world.natoms() {
                let t = uff_types[i].as_str();
                let mut req = get_reqh(&params, t);
                if charges[i] != 0.0 { req[2] = charges[i]; }
                world.nonbonded.as_mut().unwrap().reqs.as_mut_slice()[i] = req;
            }
            world.nonbonded.as_mut().unwrap().make_plqs(2.0);

            println!("=== Atom types + charges ===");
            for i in 0..world.natoms() {
                let q = world.nonbonded.as_ref().unwrap().reqs.as_slice()[i][2];
                println!("atom {:3} el {:2} type {:6} Q {:8.4}", i, elems[i], uff_types[i], q);
            }

            for ib in 0..world.uff.nbonds as usize {
                let b = world.uff.bon_atoms.as_slice()[ib];
                let ia = b[0] as usize;
                let ja = b[1] as usize;
                let a = elems[ia].as_str();
                let b = elems[ja].as_str();
                if let Some(bp) = params.get_bond_param(a, b, 1) {
                    world.uff.bon_params.as_mut_slice()[ib] = [bp.k, bp.l0];
                } else {
                    panic!("missing bond param for {}-{} order=1", a, b);
                }
            }

            for ia in 0..world.uff.nangles as usize {
                let ang = world.uff.ang_atoms.as_slice()[ia];
                let i0 = ang[0] as usize;
                let i1 = ang[1] as usize;
                let i2 = ang[2] as usize;
                let a = elems[i0].as_str();
                let b = elems[i1].as_str();
                let c = elems[i2].as_str();
                let ap = params.get_angle_param(a, b, c).unwrap_or_else(|| panic!("missing angle param for {}-{}-{}", a, b, c));

                let th0 = ap.a0.to_radians();
                let ct = th0.cos();
                let st2 = 1.0 - ct * ct;
                assert!(st2 > 1e-12, "invalid angle theta0={} deg leads to sin^2(theta0)~0", ap.a0);
                let c2 = 1.0 / (4.0 * st2);
                let c1 = -4.0 * c2 * ct;
                let c0 = c2 * (2.0 * ct * ct + 1.0);
                world.uff.ang_params.as_mut_slice()[ia] = [ap.k, c0, c1, c2, 0.0];
            }

            for id in 0..world.uff.ndihedrals as usize {
                let d = world.uff.dih_atoms.as_slice()[id];
                let a = uff_types[d.x as usize].as_str();
                let b = uff_types[d.y as usize].as_str();
                let c = uff_types[d.z as usize].as_str();
                let e = uff_types[d.w as usize].as_str();
                if let Some(dp) = params.get_dihedral_param(a, b, c, e, 1) {
                    let a0 = dp.a0.to_radians();
                    let n = dp.n as f64;
                    let phase = n * a0;
                    let s = phase.sin().abs();
                    if s > 1e-3 {
                        panic!("dihedral phase not supported by current Uff dihedral form: {}-{}-{}-{} a0={}deg n={} => n*a0={}deg", a, b, c, e, dp.a0, dp.n, phase.to_degrees());
                    }
                    let dsign = if phase.cos() >= 0.0 { 1.0 } else { -1.0 };
                    world.uff.dih_params.as_mut_slice()[id] = [dp.k, dsign, dp.n as f64];
                } else {
                    world.uff.dih_params.as_mut_slice()[id] = [0.0, 1.0, 3.0];
                }
            }

            for ii in 0..world.uff.ninversions as usize {
                let inv = world.uff.inv_atoms.as_slice()[ii];
                let ic = inv.x as usize;
                let t = uff_types[ic].as_str();
                if matches!(t, "C_R"|"C_2"|"N_R"|"O_2"|"O_R") {
                    world.uff.inv_params.as_mut_slice()[ii] = [50.0, 1.0, -1.0, 0.0];
                } else {
                    world.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0];
                }
            }
        } else {
            for i in 0..world.natoms() {
                let mut req = [1.5, 0.1, 0.0, 0.0];
                if charges[i] != 0.0 { req[2] = charges[i]; }
                world.nonbonded.as_mut().unwrap().reqs.as_mut_slice()[i] = req;
            }
            world.nonbonded.as_mut().unwrap().make_plqs(2.0);
            let apos = world.dyn_atoms.atoms.apos.as_slice();
            for ib in 0..world.uff.nbonds as usize {
                let b = world.uff.bon_atoms.as_slice()[ib];
                let ia = b[0] as usize;
                let ja = b[1] as usize;
                let d = surfmol_common::math::vec3::Vec3d::set_sub(apos[ja], apos[ia]);
                let l0 = d.norm();
                world.uff.bon_params.as_mut_slice()[ib] = [100.0, l0];
            }
        }
    }

    // Place molecule slightly above surface
    for i in 0..world.natoms() {
        world.dyn_atoms.atoms.apos.as_mut_slice()[i].z += 2.0;
    }
    world.update_hneigh();

    // Setup NaCl surface
    world.setup_nacl_surface(LATTICE_A, SURFACE_Z0 as f64, BETA_VDW, Q_AMP, PLQ_AMP);
    println!("Surface setup complete (NaCl lattice a={} Å)", LATTICE_A);

    let mut app = App::new(world, elems, params, group_size);
    println!("App initialized. Starting render loop.");
    println!("Controls: H=help  SPACE=relax  S=surface  B=bonds  P=pin  ESC=deselect");

    loop {
        app.handle_input();
        app.do_relax_step();
        if app.show_groups { app.rebuild_groups(); }
        app.draw();

        next_frame().await;
    }
}
