//! raff3d_view — interactive 3D reactive-RAFF sandbox (quaternion rigid atoms,
//! emergent bonds from `raff_reactive::pair_math`).
//!
//! Scenes are Rhai scripts (data/scenes/raff3d_*.rhai) — edit the file, press R
//! to reload, NO recompile needed. API: atom(x,y,z,"sp3"|"sp2"|"sp1"|"h") /
//! pinned(...) / random(n,seed,r,typ) (uniform sphere) / box(x0,y0,z0,x1,y1,z1,k) /
//! grid(dx) / field(bool) / a(v),b(v) param overrides.
//!
//! Field = z-slice probe-atom energy E(x,y,z=slice_z) vs all atoms.
//! Drag atoms with LMB (spring to mouse ray, k=30); pinned atoms never move.
//! Space=run/pause R=reload A=add sp3 atom B=bonds T=ports F=field G=grid cells
//! (PIC viz: faint all + blue occupied + amber stencil of selected/dragged atom,
//! thin rcut interaction lines).
//! Shift+LMB=pan RMB=orbit wheel=zoom.
//! Run: cargo run --release -p editor --bin raff3d_view [--scene file.rhai]
#![allow(deprecated)]   // winit EventLoop::run / egui Frame::none — migrate later

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use glam::{Vec2, Vec3};
use molff::raff::{self, BoxCfg, RaffConfig, RaffState, RaffTopology};
use molff::raff_reactive::{pair_math, set_reactive_inertia, ReactParams, Reactive, REACT_PEX_M};
use molrender::impostor::{AtomInstance, ImpostorRenderer};
use molrender::line_renderer::{LineRenderer, LineVertex};
use molrender::surface_renderer::SurfaceRenderer;
use molgui::gui::trackball::TrackballCam;
use numtypes::{Aabb3d, Quat4d, Vec3d, VEC3D_ZERO};
use spacc::uniform_grid::Grid3;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

const FIELD_N: u32 = 192;
const FIELD_EXT: f64 = 6.0;
const PICK_R: f32 = 0.5;
const K_SPRING: f64 = 30.0;
const E_BOND: f64 = -0.15;      // emerged-bond display threshold (well depth ~ -1)
const SCENE_DIR: &str = "data/scenes";

/// Probe atom for the field slice: point atom (no ports), r_cov=0.4.
const PROBE: ReactParams = ReactParams { r_cov: 0.4, a: 1.0, b: 0.9 };

struct Rng(u64);
impl Rng { fn next(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 11) as f64 / (1u64 << 53) as f64 } }

/// Atom flavor: port geometry + default covalent radius.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Typ { Sp3, Sp2, Sp1, H }
impl Typ {
    fn parse(s: &str) -> Self { match s { "sp3" => Self::Sp3, "sp2" => Self::Sp2, "sp1" => Self::Sp1, "h" => Self::H, _ => panic!("unknown atom type '{s}' — expected sp3|sp2|sp1|h") } }
    fn r_cov(self) -> f64 { match self { Self::Sp3 => 0.77, Self::Sp2 => 0.73, Self::Sp1 => 0.66, Self::H => 0.37 } }
    fn radius(self) -> f32 { match self { Self::H => 0.25, _ => 0.35 } }
    fn set_geo(self, topo: &mut RaffTopology, i: usize) { match self { Self::Sp3 => topo.set_sp3(i), Self::Sp2 => topo.set_sp2(i), Self::Sp1 => topo.set_sp1(i), Self::H => topo.set_1port(i, Vec3d::new(1.0, 0.0, 0.0)) } }
}

fn eatom_color(e: f64) -> [f32; 3] {
    let t = (-e / 1.5).clamp(0.0, 1.0) as f32;
    [(1.0 - t) * 0.9 + 0.1, t * 0.75 + 0.05, 0.08]
}
/// Matplotlib RdBu (11 anchors, linear interp): t=0 dark red, 0.5 white, 1 dark blue.
fn rdbu(t: f64) -> [u8; 4] {
    const C: [[f64; 3]; 11] = [
        [0.404, 0.000, 0.122], [0.698, 0.094, 0.169], [0.839, 0.376, 0.302], [0.957, 0.647, 0.510],
        [0.992, 0.859, 0.780], [0.969, 0.969, 0.969], [0.820, 0.898, 0.941], [0.569, 0.776, 0.867],
        [0.263, 0.576, 0.765], [0.129, 0.400, 0.675], [0.020, 0.188, 0.380]];
    let t = t.clamp(0.0, 1.0) * 10.0;
    let i = (t as usize).min(9);
    let f = t - i as f64;
    let rgb = [
        (C[i][0] + (C[i + 1][0] - C[i][0]) * f) as f32,
        (C[i][1] + (C[i + 1][1] - C[i][1]) * f) as f32,
        (C[i][2] + (C[i + 1][2] - C[i][2]) * f) as f32];
    [(rgb[0] * 255.0) as u8, (rgb[1] * 255.0) as u8, (rgb[2] * 255.0) as u8, 230]
}
/// E-field color: E=-EMAX (attractive) -> dark blue, E=0 -> white, E=+EMAX -> dark red.
fn field_color(e: f64) -> [u8; 4] {
    const E_MAX: f64 = 2.0;
    let t = (E_MAX - e.clamp(-E_MAX, E_MAX)) / (2.0 * E_MAX);
    rdbu(t)
}

fn ray_sphere(ro: Vec3, rd: Vec3, sc: Vec3, sr: f32) -> Option<f32> {   // main.rs:62
    let oc = ro - sc; let b = oc.dot(rd); let c = oc.dot(oc) - sr * sr; let disc = b * b - c;
    if disc < 0.0 { return None; } let t = -b - disc.sqrt(); if t >= 0.0 { Some(t) } else { None }
}
/// Perpendicular spring toward the mouse ray (FireCore getForceSpringRay, main.rs:67).
fn spring_ray(p: Vec3d, ro: Vec3d, rd: Vec3d, k: f64) -> Vec3d {
    let dp = p - ro; let cdot = rd.dot(dp); let dp_perp = dp - rd * cdot; dp_perp * (-k)
}

/// 12 edges of AABB [lo,hi] as line vertices (edge topology via spacc::aabb::aabb_edges).
fn push_box(lines: &mut Vec<LineVertex>, lo: [f64; 3], hi: [f64; 3], col: [f32; 4]) {
    let e = spacc::aabb::aabb_edges(Aabb3d::new(Vec3d::new(lo[0], lo[1], lo[2]), Vec3d::new(hi[0], hi[1], hi[2])));
    for k in 0..12 { lines.push(LineVertex { pos: e[k * 2], col }); lines.push(LineVertex { pos: e[k * 2 + 1], col }); }
}
/// Wireframe of grid cell `c`: lo = origin + unravel(c)*dx, hi = lo + dx.
fn cell_box(lines: &mut Vec<LineVertex>, g: &Grid3, c: usize, col: [f32; 4]) {
    let [ix, iy, iz] = g.unravel(c);
    let lo = [g.origin[0] + ix as f64 * g.dx, g.origin[1] + iy as f64 * g.dx, g.origin[2] + iz as f64 * g.dx];
    push_box(lines, lo, [lo[0] + g.dx, lo[1] + g.dx, lo[2] + g.dx], col);
}

// ---------------- Rhai scene config ----------------

#[derive(Default, Clone)]
struct SceneCfg {
    atoms: Vec<(Vec3d, Typ, bool)>,     // pos, type, pinned
    box_cfg: Option<BoxCfg>,
    grid_dx: Option<f64>,
    a: Option<f64>, b: Option<f64>,
    field: Option<bool>,
}

/// Run a .rhai scene script; registered fns fill SceneCfg.
fn load_scene(path: &std::path::Path) -> SceneCfg {
    let cfg = Rc::new(RefCell::new(SceneCfg::default()));
    let mut engine = rhai::Engine::new();
    {
        let c = cfg.clone();
        engine.register_fn("atom", move |x: f64, y: f64, z: f64, t: &str| c.borrow_mut().atoms.push((Vec3d::new(x, y, z), Typ::parse(t), false)));
    }
    {
        let c = cfg.clone();
        engine.register_fn("pinned", move |x: f64, y: f64, z: f64, t: &str| c.borrow_mut().atoms.push((Vec3d::new(x, y, z), Typ::parse(t), true)));
    }
    {
        let c = cfg.clone();
        engine.register_fn("random", move |n: i64, seed: i64, r: f64, t: &str| {
            let typ = Typ::parse(t);
            let mut rng = Rng((seed as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1);
            for _ in 0..n {   // uniform in sphere: dir from (cos θ, φ), radius r·u^(1/3)
                let cz = 2.0 * rng.next() - 1.0;
                let sz = (1.0 - cz * cz).max(0.0).sqrt();
                let ph = rng.next() * std::f64::consts::TAU;
                let rr = r * rng.next().cbrt();
                c.borrow_mut().atoms.push((Vec3d::new(rr * sz * ph.cos(), rr * sz * ph.sin(), rr * cz), typ, false));
            }
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("box", move |x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64, k: f64| {
            c.borrow_mut().box_cfg = Some(BoxCfg { enabled: true, min: Vec3d::new(x0, y0, z0), max: Vec3d::new(x1, y1, z1), k });
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("grid", move |dx: f64| c.borrow_mut().grid_dx = Some(dx));
    }
    {
        let c = cfg.clone();
        engine.register_fn("field", move |v: bool| c.borrow_mut().field = Some(v));
    }
    {
        let c = cfg.clone();
        engine.register_fn("a", move |v: f64| c.borrow_mut().a = Some(v));
    }
    {
        let c = cfg.clone();
        engine.register_fn("b", move |v: f64| c.borrow_mut().b = Some(v));
    }
    match engine.eval_file::<rhai::Dynamic>(path.to_path_buf()) {
        Ok(_) => {}
        Err(e) => panic!("rhai scene {} failed: {e}", path.display()),
    }
    let out = cfg.borrow().clone();
    out
}

/// Build (state, topo, params, pinned, types) from a scene cfg.
/// Pinned atoms get mass=1e12 + inv_inertia=0; velocities zeroed each step.
/// Quaternion rotating unit vector a onto b (shortest arc).
fn quat_align(a: Vec3d, b: Vec3d) -> Quat4d {
    let ax = Vec3d::cross(a, b);
    let s = ax.norm();
    let c = a.dot(b);
    if s < 1e-12 {
        if c > 0.0 { return Quat4d::new(0.0, 0.0, 0.0, 1.0); }
        // a ≈ -b: 180° about any axis ⊥ a — 180° about a itself leaves it invariant!
        let mut ax = Vec3d::cross(a, if a.x.abs() < 0.9 { Vec3d::new(1.0, 0.0, 0.0) } else { Vec3d::new(0.0, 1.0, 0.0) });
        ax.normalize();
        return Quat4d::new(ax.x, ax.y, ax.z, 0.0);
    }
    raff::quat_from_omega_dt(ax * (s.atan2(c) / s), 1.0)
}

fn build_sys(cfg: &SceneCfg) -> (RaffState, RaffTopology, Vec<ReactParams>, Vec<bool>, Vec<Typ>) {
    let n = cfg.atoms.len();
    assert!(n > 0, "scene has no atoms");
    let mut topo = RaffTopology::new(n);
    let mut params = Vec::with_capacity(n);
    let (mut pinned, mut types) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut state = RaffState::new(n);
    for (i, &(p, t, pin)) in cfg.atoms.iter().enumerate() {
        t.set_geo(&mut topo, i);
        params.push(ReactParams { r_cov: t.r_cov(), a: cfg.a.unwrap_or(1.0), b: cfg.b.unwrap_or(0.9) });
        pinned.push(pin); types.push(t);
        state.pos[i] = p;   // identity quats — ports align under torque
    }
    // 1-port H: aim port at the nearest atom — a port pointing fully away is
    // outside the gate ⇒ zero torque ⇒ it can never reorient (rigid-port dead
    // zone). Multi-port atoms always catch some torque so identity is fine.
    for i in 0..n {
        if types[i] != Typ::H { continue; }
        let mut best = (usize::MAX, f64::MAX);
        for j in 0..n { if j != i { let d2 = (state.pos[j] - state.pos[i]).norm2(); if d2 < best.1 { best = (j, d2); } } }
        if best.0 != usize::MAX {
            let mut d = state.pos[best.0] - state.pos[i]; d.normalize();
            state.quat[i] = quat_align(Vec3d::new(1.0, 0.0, 0.0), d);
        }
    }
    set_reactive_inertia(&mut topo, &params);
    for (i, &pin) in pinned.iter().enumerate() { if pin { topo.mass[i] = 1e12; topo.inv_inertia[i] = 0.0; } }
    (state, topo, params, pinned, types)
}

struct App {
    window: Arc<Window>,
    state: RaffState, topo: RaffTopology, rx: Reactive, cfg: RaffConfig,
    fapos: Vec<Vec3d>, tau: Vec<Vec3d>,
    pinned: Vec<bool>, types: Vec<Typ>,
    bonds: Vec<(u32, u32, f64)>, etot: f64, eval_us: f64,
    scene_path: PathBuf, scene_files: Vec<PathBuf>,
    run: bool, dt: f64, cdamp: f64, per_frame: usize,
    use_grid: bool, grid_dx: f64,
    pa: f64, pb: f64, pex_m: u32,
    selected: Option<usize>, dragging: Option<usize>,
    show_field: bool, show_bonds: bool, show_ports: bool, show_grid: bool,
    slice_z: f64, fwd_scratch: Vec<usize>, sel_n: usize, sel_cells: usize,
    cam: TrackballCam,
    renderer: ImpostorRenderer, instances: Vec<AtomInstance>,
    line_renderer: LineRenderer, surface_renderer: SurfaceRenderer,
    field_tex: wgpu::Texture, field_dirty: bool,
    device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface<'static>,
    egui_ctx: egui::Context, egui_state: egui_winit::State, egui_renderer: egui_wgpu::Renderer,
    mouse_now: Vec2, lmb_down: bool, trackballing: bool, trackball_prev: Vec2, shift: bool,
    window_size: (f32, f32),
}

impl App {
    fn new(window: Arc<Window>, scene_path: PathBuf) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(), display: None });
        let surface = instance.create_surface(window.clone()).expect("create_surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("no adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("no device");
        let device = Arc::new(device); let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: caps.present_modes[0], alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2 };
        surface.configure(&device, &config);
        let (ww, wh) = (size.width as f32, size.height as f32);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(egui_ctx.clone(), egui_ctx.viewport_id(), &window, Some(window.scale_factor() as f32), None, None);
        let egui_renderer = egui_wgpu::Renderer::new(&*device, config.format, egui_wgpu::RendererOptions::default());
        let line_renderer = LineRenderer::new(device.clone(), queue.clone(), config.format);
        let surface_renderer = SurfaceRenderer::new(device.clone(), queue.clone(), config.format);
        let renderer = ImpostorRenderer::new(device.clone(), queue.clone(), 4096, config.format);
        let field_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_tex"), size: wgpu::Extent3d { width: FIELD_N, height: FIELD_N, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let scene_files = Self::list_scenes();
        let scfg = load_scene(&scene_path);
        let (state, topo, params, pinned, types) = build_sys(&scfg);
        let natoms = state.natoms;
        let mut rx = Reactive::new(&topo, params, REACT_PEX_M);
        if let Some(dx) = scfg.grid_dx { let (o, n) = grid_domain(&state, scfg.box_cfg.as_ref(), rx.cfg.rcut, dx); rx.set_grid(o, n, dx); }
        let cfg = RaffConfig { dt: 0.02, cdamp: 0.9, rot_damp: 0.9, flim: 20.0, box_cfg: scfg.box_cfg.unwrap_or_default(), ..Default::default() };
        let mut app = Self {
            window, state, topo, rx, cfg,
            fapos: vec![VEC3D_ZERO; natoms], tau: vec![VEC3D_ZERO; natoms],
            pinned, types,
            bonds: Vec::new(), etot: 0.0, eval_us: 0.0,
            scene_path, scene_files,
            run: true, dt: 0.02, cdamp: 0.9, per_frame: 40,
            use_grid: scfg.grid_dx.is_some(), grid_dx: scfg.grid_dx.unwrap_or(2.0),
            pa: scfg.a.unwrap_or(1.0), pb: scfg.b.unwrap_or(0.9), pex_m: REACT_PEX_M,
            selected: None, dragging: None,
            show_field: scfg.field.unwrap_or(true), show_bonds: true, show_ports: true, show_grid: scfg.grid_dx.is_some(),
            slice_z: 0.0, fwd_scratch: Vec::with_capacity(343), sel_n: 0, sel_cells: 0,   // 343 = (2s+1)^3 for s<=3
            cam: TrackballCam::new(Vec3::new(0.0, 0.0, 0.0), 12.0),
            renderer, instances: Vec::new(), line_renderer, surface_renderer,
            field_tex, field_dirty: true,
            device, queue, config, surface,
            egui_ctx, egui_state, egui_renderer,
            mouse_now: Vec2::ZERO, lmb_down: false, trackballing: false, trackball_prev: Vec2::ZERO, shift: false,
            window_size: (ww, wh),
        };
        app.eval_once();
        app
    }

    fn list_scenes() -> Vec<PathBuf> {
        let dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../../..").join(SCENE_DIR);
        let mut v: Vec<PathBuf> = std::fs::read_dir(&dir).map(|rd| rd.flatten().map(|e| e.path())
            .filter(|p| p.file_name().map(|f| f.to_string_lossy().starts_with("raff3d_") && f.to_string_lossy().ends_with(".rhai")).unwrap_or(false))
            .collect()).unwrap_or_default();
        v.sort();
        v
    }

    fn reload_scene(&mut self) {
        let scfg = load_scene(&self.scene_path);
        let (state, topo, params, pinned, types) = build_sys(&scfg);
        self.state = state; self.topo = topo; self.pinned = pinned; self.types = types;
        self.rx = Reactive::new(&self.topo, params, self.pex_m);
        self.fapos = vec![VEC3D_ZERO; self.state.natoms]; self.tau = vec![VEC3D_ZERO; self.state.natoms];
        self.cfg.box_cfg = scfg.box_cfg.unwrap_or_default();
        self.use_grid = scfg.grid_dx.is_some();
        self.grid_dx = scfg.grid_dx.unwrap_or(self.grid_dx);
        if let Some(dx) = scfg.grid_dx { let (o, n) = grid_domain(&self.state, scfg.box_cfg.as_ref(), self.rx.cfg.rcut, dx); self.rx.set_grid(o, n, dx); }
        self.pa = scfg.a.unwrap_or(self.pa); self.pb = scfg.b.unwrap_or(self.pb);
        self.show_field = scfg.field.unwrap_or(self.show_field);
        self.dragging = None; self.selected = None;
        self.eval_once(); self.field_dirty = true;
        println!("reloaded scene {:?}: {} atoms", self.scene_path.file_name(), self.state.natoms);
    }

    /// Grid domain: atom bbox ∪ box corners, expanded by rcut + dx margin.
    fn apply_params(&mut self) {
        for i in 0..self.rx.params.len() { self.rx.params[i].a = self.pa; self.rx.params[i].b = self.pb; }
        let params = self.rx.params.clone();
        self.rx.reconfigure(params, self.pex_m);   // recomputes rcut, drops grid
        if self.use_grid {
            let (o, n) = grid_domain(&self.state, if self.cfg.box_cfg.enabled { Some(&self.cfg.box_cfg) } else { None }, self.rx.cfg.rcut, self.grid_dx);
            self.rx.set_grid(o, n, self.grid_dx);
        }
        self.eval_once();
        self.field_dirty = true;
    }

    fn eval_once(&mut self) {
        let t = std::time::Instant::now();
        let e = self.rx.eval_reactive(&self.state, &self.topo, &mut self.fapos, &mut self.tau);
        self.etot = e + raff::eval_box_forces(&self.state, &self.cfg.box_cfg, &mut self.fapos);
        self.eval_us = self.eval_us * 0.9 + t.elapsed().as_secs_f64() * 1e6 * 0.1;
    }

    /// Ray ∩ plane through `p0` perpendicular to view dir (for add-atom target).
    fn mouse_plane(&self, m: Vec2, p0: Vec3) -> Vec3d {
        let (ro, rd) = self.cam.screen_ray(m, self.window_size.0, self.window_size.1);
        let t = (p0 - ro).dot(rd);
        let p = ro + rd * t;
        Vec3d::new(p.x as f64, p.y as f64, p.z as f64)
    }

    fn pick_atom(&self, m: Vec2) -> Option<usize> {
        let (ro, rd) = self.cam.screen_ray(m, self.window_size.0, self.window_size.1);
        let mut best: Option<(usize, f32)> = None;
        for i in 0..self.state.natoms {
            let p = self.state.pos[i];
            if let Some(t) = ray_sphere(ro, rd, Vec3::new(p.x as f32, p.y as f32, p.z as f32), PICK_R) {
                if best.map(|(_, tb)| t < tb).unwrap_or(true) { best = Some((i, t)); }
            }
        }
        best.map(|(i, _)| i)
    }

    /// per_frame substeps: eval_reactive + box + drag spring → integrate_md.
    /// Always the eval+integrate path (spring is a no-op when not dragging).
    fn step(&mut self) {
        for _ in 0..self.per_frame {
            let t = std::time::Instant::now();
            let mut e = self.rx.eval_reactive(&self.state, &self.topo, &mut self.fapos, &mut self.tau);
            e += raff::eval_box_forces(&self.state, &self.cfg.box_cfg, &mut self.fapos);
            if let Some(i) = self.dragging {
                let (ro, rd) = self.cam.screen_ray(self.mouse_now, self.window_size.0, self.window_size.1);
                let ro3 = Vec3d::new(ro.x as f64, ro.y as f64, ro.z as f64);
                let rd3 = Vec3d::new(rd.x as f64, rd.y as f64, rd.z as f64);
                self.fapos[i].add(spring_ray(self.state.pos[i], ro3, rd3, K_SPRING));
                self.state.vel[i] = VEC3D_ZERO; self.state.omega[i] = VEC3D_ZERO;
            }
            for i in 0..self.state.natoms { if self.pinned[i] { self.state.vel[i] = VEC3D_ZERO; self.state.omega[i] = VEC3D_ZERO; } }
            raff::integrate_md(&mut self.state, &self.topo, &self.cfg, &self.fapos, &self.tau);
            self.etot = e;
            self.eval_us = self.eval_us * 0.95 + t.elapsed().as_secs_f64() * 1e6 * 0.05;
        }
        self.field_dirty = true;
    }

    /// Probe-atom energy on the z=slice_z plane via pair_math (point probe: hj=[]).
    fn probe_e(&self, p: Vec3d, skip: usize) -> f64 {
        let mut e = 0.0;
        for i in 0..self.topo.natoms {
            if i == skip { continue; }
            let np = self.topo.nport[i] as usize;
            if let Some((ep, _, _, _)) = pair_math(&self.rx.cfg, &self.rx.params[i], &self.rx.h_world[i * 4..i * 4 + np], self.state.pos[i], &PROBE, &[], p) { e += ep; }
        }
        e
    }

    fn update_field(&mut self) {
        if !self.field_dirty { return; }
        self.field_dirty = false;
        let skip = self.dragging.or(self.selected).unwrap_or(usize::MAX);
        let n = FIELD_N as usize;
        let z = self.slice_z;
        let mut buf = vec![0u8; n * n * 4];
        for iy in 0..n {
            let y = -FIELD_EXT + 2.0 * FIELD_EXT * (iy as f64 + 0.5) / n as f64;
            for ix in 0..n {
                let x = -FIELD_EXT + 2.0 * FIELD_EXT * (ix as f64 + 0.5) / n as f64;
                let c = field_color(self.probe_e(Vec3d::new(x, y, z), skip));
                buf[(iy * n + ix) * 4..(iy * n + ix) * 4 + 4].copy_from_slice(&c);
            }
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.field_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &buf, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * FIELD_N), rows_per_image: Some(FIELD_N) },
            wgpu::Extent3d { width: FIELD_N, height: FIELD_N, depth_or_array_layers: 1 });
    }

    fn collect_lines(&mut self) -> Vec<LineVertex> {
        let mut lines = Vec::new();
        let lv = |x: f32, y: f32, z: f32, c: [f32; 4]| LineVertex { pos: [x, y, z], col: c };
        let pv = |p: Vec3d| [p.x as f32, p.y as f32, p.z as f32];
        if self.show_bonds {
            self.bonds = self.rx.emerged_bonds(&self.state, &self.topo, E_BOND);
            for &(i, j, ep) in &self.bonds {
                let a = ((-ep) as f32).clamp(0.2, 1.0);
                let (p, q) = (pv(self.state.pos[i as usize]), pv(self.state.pos[j as usize]));
                lines.push(lv(p[0], p[1], p[2], [a, a, a, 1.0])); lines.push(lv(q[0], q[1], q[2], [a, a, a, 1.0]));
            }
        }
        if self.show_ports {
            for i in 0..self.topo.natoms {
                let np = self.topo.nport[i] as usize;
                for s in 0..np {
                    let p = pv(self.state.pos[i]);
                    let h = self.rx.h_world[i * 4 + s] * 0.5 + self.state.pos[i];
                    let q = pv(h);
                    lines.push(lv(p[0], p[1], p[2], [0.2, 1.0, 0.4, 0.9])); lines.push(lv(q[0], q[1], q[2], [0.2, 1.0, 0.4, 0.9]));
                }
                if self.pinned[i] {   // pinned marker: dark cross
                    let p = pv(self.state.pos[i]);
                    lines.push(lv(p[0] - 0.15, p[1], p[2], [0.1, 0.1, 0.1, 1.0])); lines.push(lv(p[0] + 0.15, p[1], p[2], [0.1, 0.1, 0.1, 1.0]));
                    lines.push(lv(p[0], p[1] - 0.15, p[2], [0.1, 0.1, 0.1, 1.0])); lines.push(lv(p[0], p[1] + 0.15, p[2], [0.1, 0.1, 0.1, 1.0]));
                }
            }
        }
        if self.show_grid {
            if let Some(g) = self.rx.grid.as_ref() {   // PIC viz: cell wireframes + domain box
                for c in 0..g.buckets.ncells { cell_box(&mut lines, g, c, [0.35, 0.35, 0.45, 0.18]); }   // all cells faint
                for &c in g.cell_of.iter().collect::<std::collections::HashSet<_>>() {   // occupied cells stronger blue
                    cell_box(&mut lines, g, c as usize, [0.2, 0.5, 0.9, 0.8]);
                }
                let lo = g.origin;   // grid domain wireframe
                push_box(&mut lines, lo, [lo[0] + g.n[0] as f64 * g.dx, lo[1] + g.n[1] as f64 * g.dx, lo[2] + g.n[2] as f64 * g.dx], [0.35, 0.45, 0.9, 0.7]);
            }
        }
        // --- selected/dragged atom: amber stencil cells + thin rcut interaction lines ---
        self.sel_n = 0; self.sel_cells = 0;
        if let Some(i) = self.selected.or(self.dragging) {
            if self.show_grid {
                if let Some(g) = self.rx.grid.as_ref() {   // own cell + forward-stencil cells (amber)
                    let c = g.cell_of[i] as usize;
                    let s = (self.rx.cfg.rcut / g.dx).ceil() as i64;
                    g.forward_cells(c, s, &mut self.fwd_scratch);
                    cell_box(&mut lines, g, c, [0.95, 0.7, 0.1, 0.85]);
                    self.sel_cells = self.fwd_scratch.len() + 1;
                    for k in 0..self.fwd_scratch.len() { let cc = self.fwd_scratch[k]; cell_box(&mut lines, g, cc, [0.95, 0.7, 0.1, 0.4]); }
                }
            }
            let rc2 = self.rx.cfg.rcut * self.rx.cfg.rcut;   // lines to every atom within rcut
            let (pi, npi, pos_i) = (self.rx.params[i], self.topo.nport[i] as usize, self.state.pos[i]);
            for j in 0..self.state.natoms {
                if j == i { continue; }
                let pos_j = self.state.pos[j];
                if (pos_j - pos_i).norm2() > rc2 { continue; }
                self.sel_n += 1;
                let npj = self.topo.nport[j] as usize;
                let ep = pair_math(&self.rx.cfg, &pi, &self.rx.h_world[i * 4..i * 4 + npi], pos_i, &self.rx.params[j], &self.rx.h_world[j * 4..j * 4 + npj], pos_j).map(|t| t.0).unwrap_or(0.0);
                let col = if ep < -0.05 { [0.2, 0.9, 1.0, (0.25 + 0.6 * (-ep) as f32).min(1.0)] }   // attractive: cyan, alpha ~ strength
                        else { [0.95, 0.45, 0.35, 0.2] };                                          // repulsive/none: faint red
                let (p, q) = (pv(pos_i), pv(pos_j));
                lines.push(lv(p[0], p[1], p[2], col)); lines.push(lv(q[0], q[1], q[2], col));
            }
        }
        if self.cfg.box_cfg.enabled {   // box wireframe
            let (lo, hi) = (self.cfg.box_cfg.min, self.cfg.box_cfg.max);
            push_box(&mut lines, [lo.x, lo.y, lo.z], [hi.x, hi.y, hi.z], [0.9, 0.6, 0.1, 0.7]);
        }
        lines
    }

    fn update(&mut self, event: &WindowEvent, egui_consumed: bool) {
        match event {
            WindowEvent::Resized(sz) => {
                self.window_size = (sz.width as f32, sz.height as f32);
                self.config.width = sz.width.max(1); self.config.height = sz.height.max(1);
                self.surface.configure(&self.device, &self.config);
                self.renderer.set_target_size(self.config.width, self.config.height);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let m = Vec2::new(position.x as f32, position.y as f32);
                if self.trackballing { self.cam.rotate(self.trackball_prev, m, self.window_size.0, self.window_size.1); self.trackball_prev = m; }
                if self.lmb_down && self.shift && self.dragging.is_none() { let d = m - self.mouse_now; self.cam.pan(d.x, d.y); }
                self.mouse_now = m;
                if self.dragging.is_some() { self.field_dirty = true; }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                match state {
                    ElementState::Pressed if !egui_consumed => {
                        self.lmb_down = true;
                        if !self.shift {
                            if let Some(i) = self.pick_atom(self.mouse_now) {
                                if !self.pinned[i] { self.dragging = Some(i); }
                                self.selected = Some(i);
                            }
                        }
                    }
                    ElementState::Released => {
                        self.lmb_down = false;
                        if self.dragging.is_none() && !egui_consumed { self.selected = self.pick_atom(self.mouse_now); }
                        self.dragging = None;
                        self.field_dirty = true;
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                self.trackballing = *state == ElementState::Pressed;
                self.trackball_prev = self.mouse_now;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta { MouseScrollDelta::LineDelta(_, y) => *y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01 };
                self.cam.zoom(dy);
            }
            WindowEvent::ModifiersChanged(m) => { self.shift = m.state().shift_key(); }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Named(NamedKey::Space), state: ElementState::Pressed, .. }, .. } => { self.run = !self.run; }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Character(c), state: ElementState::Pressed, .. }, .. } if !egui_consumed => {
                match c.as_str() {
                    "f" | "F" => { self.show_field = !self.show_field; }
                    "b" | "B" => { self.show_bonds = !self.show_bonds; }
                    "t" | "T" => { self.show_ports = !self.show_ports; }
                    "g" | "G" => { self.show_grid = !self.show_grid; }
                    "r" | "R" => { self.scene_files = Self::list_scenes(); self.reload_scene(); }
                    "a" | "A" => {   // add sp3 atom at ray ∩ plane through cam target ⊥ view
                        let p = self.mouse_plane(self.mouse_now, self.cam.target);
                        let i = self.state.natoms;
                        self.topo.natoms += 1;
                        self.topo.nport.push(4); self.topo.port_local.resize(i * 4 + 4, VEC3D_ZERO);
                        self.topo.neighs.push(numtypes::Quat4i::new(-1, -1, -1, -1)); self.topo.neigh_bs.push(numtypes::Quat4i::new(-1, -1, -1, -1));
                        self.topo.mass.push(1.0); self.topo.inv_inertia.push(0.0);
                        self.topo.nb_params.push(Default::default()); self.topo.excl.resize(self.topo.excl.len() + 16, -1);
                        self.topo.set_sp3(i);
                        self.state.natoms += 1;
                        self.state.pos.push(p); self.state.quat.push(Quat4d::new(0.0, 0.0, 0.0, 1.0));
                        self.state.vel.push(VEC3D_ZERO); self.state.omega.push(VEC3D_ZERO);
                        self.fapos.push(VEC3D_ZERO); self.tau.push(VEC3D_ZERO);
                        self.pinned.push(false); self.types.push(Typ::Sp3);
                        self.rx.params.push(ReactParams { r_cov: Typ::Sp3.r_cov(), a: self.pa, b: self.pb });
                        self.rx.h_world.resize(self.state.natoms * 4, VEC3D_ZERO);
                        self.rx.eatom.push(0.0);
                        self.topo.inv_inertia[i] = 1.0 / (0.4 * (2.0 * Typ::Sp3.r_cov()).powi(2) + 1e-18);   // set_reactive_inertia formula
                        self.rx.grid = None; self.apply_params();   // natom changed → rebuild grid + rcut
                        println!("added atom {i} at ({:.2},{:.2},{:.2})", p.x, p.y, p.z);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn prepare(&mut self) {
        if self.run { self.step(); }
        self.instances = (0..self.state.natoms).map(|i| AtomInstance {
            pos: [self.state.pos[i].x as f32, self.state.pos[i].y as f32, self.state.pos[i].z as f32],
            radius: self.types[i].radius(),
            color: if self.dragging == Some(i) { [1.0, 0.5, 0.9] }
                   else if self.pinned[i] { [0.15, 0.15, 0.15] }
                   else if self.selected == Some(i) { [1.0, 0.8, 0.2] }
                   else { eatom_color(self.rx.eatom[i]) }, _pad: 0.0,
        }).collect();
        self.renderer.set_atoms(&self.instances);
        if self.show_field { self.update_field(); }
    }

    fn render(&mut self) {
        let frame = self.surface.get_current_texture();
        let surface_texture = match frame {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => { self.surface.configure(&self.device, &self.config); return; }
            _ => return,
        };
        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let cam = self.cam.camera_data(self.config.width, self.config.height);
        self.renderer.render(&view, wgpu::Color { r: 0.05, g: 0.05, b: 0.07, a: 1.0 }, &cam);

        if self.show_field {
            let tex_view = self.field_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let e = FIELD_EXT as f32; let z = self.slice_z as f32;
            self.surface_renderer.render(&mut encoder, &view, self.renderer.depth_view(), &cam, &tex_view,
                [-e, -e, z], [2.0 * e, 0.0, 0.0], [0.0, 2.0 * e, 0.0]);
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        let lines = self.collect_lines();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.line_renderer.render(&mut encoder, &view, self.renderer.depth_view(), &cam, &lines);

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let egui_ctx = self.egui_ctx.clone();
        let full_output = egui_ctx.run(raw_input, |ctx| { self.draw_egui(ctx); });
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, d) in &full_output.textures_delta.set { self.egui_renderer.update_texture(&*self.device, &self.queue, *id, d); }
        for id in &full_output.textures_delta.free { self.egui_renderer.free_texture(id); }
        let sd = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.config.width, self.config.height], pixels_per_point: full_output.pixels_per_point };
        self.egui_renderer.update_buffers(&*self.device, &self.queue, &mut encoder, &tris, &sd);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None, occlusion_query_set: None, timestamp_writes: None, multiview_mask: None,
            });
            self.egui_renderer.render(&mut pass.forget_lifetime(), &tris, &sd);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }

    fn draw_egui(&mut self, ctx: &egui::Context) {
        let mut params_changed = false;
        egui::Window::new("raff3d").anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
            .frame(egui::Frame::none().fill(egui::Color32::from_rgba_premultiplied(0, 0, 0, 190)))
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(format!("scene: {}", self.scene_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())).size(15.0).color(egui::Color32::WHITE));
                let mut pick: Option<usize> = None;
                ui.horizontal(|ui| {
                    for (i, p) in self.scene_files.iter().enumerate() {
                        if ui.small_button(p.file_stem().map(|s| s.to_string_lossy().replace("raff3d_", "")).unwrap_or_default()).clicked() { pick = Some(i); }
                    }
                    if ui.small_button("reload [R]").clicked() { pick = Some(usize::MAX); }
                });
                match pick {
                    Some(i) if i < self.scene_files.len() => { self.scene_path = self.scene_files[i].clone(); self.reload_scene(); }
                    Some(_) => { self.scene_files = Self::list_scenes(); self.reload_scene(); }
                    None => {}
                }
                ui.label(egui::RichText::new(format!("E_total = {:9.4} eV   eval = {:.1} us/step", self.etot, self.eval_us)).size(13.0));
                ui.label(egui::RichText::new(format!("rcut = r0+2^{}/b = {:.2} A (auto from pex_m)", self.rx.cfg.pex_m, self.rx.cfg.rcut)).size(11.0));
                if let Some(i) = self.selected {
                    let q = self.state.quat[i];
                    ui.label(egui::RichText::new(format!("atom {i} {:?}: E={:.3} nport={} quat=({:.2},{:.2},{:.2},{:.2}){}", self.types[i], self.rx.eatom[i], self.topo.nport[i], q.x, q.y, q.z, q.w, if self.pinned[i] { " [pinned]" } else { "" })).size(12.0).color(egui::Color32::YELLOW));
                    ui.label(egui::RichText::new(format!("  interacts with {} atoms ({} cells in stencil)", self.sel_n, self.sel_cells)).size(11.0).color(egui::Color32::LIGHT_BLUE));
                }
                ui.separator();
                ui.label("pair potential");
                params_changed |= ui.add(egui::Slider::new(&mut self.pa, 0.0..=3.0).text("a (depth)")).changed();
                params_changed |= ui.add(egui::Slider::new(&mut self.pb, 0.3..=3.0).text("b (steepness)")).changed();
                params_changed |= ui.add(egui::Slider::new(&mut self.pex_m, 1..=6).text("pex m (cutoff: r0+2^m/b)")).changed();
                params_changed |= ui.checkbox(&mut self.rx.cfg.use_cij, "use_cij (h_i.h_j gate)").changed();
                ui.separator();
                ui.label("solver / neighbors");
                params_changed |= ui.add(egui::Slider::new(&mut self.dt, 0.005..=0.05).text("dt")).changed();
                params_changed |= ui.add(egui::Slider::new(&mut self.cdamp, 0.7..=0.999).text("damping")).changed();
                ui.add(egui::Slider::new(&mut self.per_frame, 1..=400).text("steps/frame"));
                params_changed |= ui.checkbox(&mut self.use_grid, "use Grid3").changed();
                if self.use_grid { params_changed |= ui.add(egui::Slider::new(&mut self.grid_dx, 0.5..=4.0).text("grid dx [A]")).changed(); }
                if ui.add(egui::Slider::new(&mut self.slice_z, -4.0..=4.0).text("field slice z")).changed() { self.field_dirty = true; }
                ui.separator();
                ui.label(egui::RichText::new("LMB drag · Shift+LMB pan · RMB orbit · wheel zoom\nSpace run · R reload · A add sp3 · B bonds · T ports · F field · G grid").size(10.5).color(egui::Color32::GRAY));
            });
        if params_changed {
            self.cfg.dt = self.dt; self.cfg.cdamp = self.cdamp; self.cfg.rot_damp = self.cdamp;
            self.apply_params();
        }
    }
}

/// Grid domain covering atoms (+ box) expanded by rcut + dx margin.
fn grid_domain(state: &RaffState, box_cfg: Option<&BoxCfg>, rcut: f64, dx: f64) -> ([f64; 3], [usize; 3]) {
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &state.pos { for k in 0..3 { let c = [p.x, p.y, p.z][k]; lo[k] = lo[k].min(c); hi[k] = hi[k].max(c); } }
    if let Some(b) = box_cfg {
        for k in 0..3 { lo[k] = lo[k].min([b.min.x, b.min.y, b.min.z][k]); hi[k] = hi[k].max([b.max.x, b.max.y, b.max.z][k]); }
    }
    let m = rcut + dx;
    let o = [lo[0] - m, lo[1] - m, lo[2] - m];
    let n = [0, 1, 2].map(|k| ((hi[k] - lo[k] + 2.0 * m) / dx).ceil().max(1.0) as usize);
    (o, n)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../../..");
    // --scene accepts a bare filename (resolved in data/scenes/) or a path
    let scene = args.windows(2).find(|w| w[0] == "--scene").map(|w| {
        let p = PathBuf::from(&w[1]);
        if p.is_file() { p } else { let q = root.join(SCENE_DIR).join(&p); assert!(q.is_file(), "scene not found: {} (tried {})", p.display(), q.display()); q }
    }).unwrap_or_else(|| root.join(SCENE_DIR).join("raff3d_methane.rhai"));
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title(format!("raff3d_view — {}", scene.display())).with_inner_size(winit::dpi::PhysicalSize::new(1400, 900))).unwrap());
    let mut app = App::new(window.clone(), scene);
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                app.cam.update(1.0 / 60.0);
                app.prepare();
                app.render();
            }
            Event::WindowEvent { event, .. } => {
                let r = app.egui_state.on_window_event(&app.window, &event);
                app.update(&event, r.consumed);
                app.window.request_redraw();
            }
            Event::AboutToWait => { app.window.request_redraw(); }
            _ => {}
        }
    }).unwrap();
}
