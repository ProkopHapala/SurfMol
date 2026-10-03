//! rarff2d_view — interactive 2D rod-free reactive-FF sandbox.
//!
//! Scenes are Rhai scripts (data/scenes/*.rhai) — edit the file, press R to
//! reload, NO recompile needed. API: atom(x,y,phi_deg) / pinned(x,y,phi_deg) /
//! random(n,seed,r) / ring(x,y,nfold,R) / ring_full(...) / arena(r,k) / grid(dx) /
//! field(bool) / w(v),a(v),b(v) overrides.
//!
//! Ground plane = probe-atom potential E(x,y) of all OTHER atoms + rings.
//! Drag atoms with LMB; pinned atoms (black cross) never move.
//! Space=run/pause R=reload scene A=add atom N=random F=field B=bonds T=ports
//! E=rings G=grid cells LMB+Shift=pan RMB=orbit wheel=zoom.
//! Run: cargo run --release -p editor --bin rarff2d_view [--scene file.rhai]
#![allow(deprecated)]   // winit EventLoop::run / egui Frame::none — migrate later

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use glam::{Vec2, Vec3};
use molff::rarff2d::{gate, Rarff2d, Rarff2dType, Ring2d, TYPE_SP2};
use molrender::impostor::{AtomInstance, ImpostorRenderer};
use molrender::line_renderer::{LineRenderer, LineVertex};
use molrender::surface_renderer::SurfaceRenderer;
use molgui::gui::trackball::TrackballCam;
use numtypes::Vec2d;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

const FIELD_N: u32 = 192;
const FIELD_EXT: f64 = 8.0;
const FIELD_Z: f32 = -0.55;
const PICK_R: f64 = 0.45;
const BOND_R: f64 = 1.6;
const GATE_Y_MIN: f64 = 0.3;
const SCENE_DIR: &str = "data/scenes";

struct Rng(u64);
impl Rng { fn next(&mut self) -> f64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 11) as f64 / (1u64 << 53) as f64 } }

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

/// E-field color: E=-EMAX (attractive) -> dark blue, E=0 -> white, E=+EMAX -> dark red (RdBu_r orientation).
fn field_color(e: f64) -> [u8; 4] {
    const E_MAX: f64 = 2.0;
    let t = (E_MAX - e.clamp(-E_MAX, E_MAX)) / (2.0 * E_MAX);
    rdbu(t)
}

// ---------------- Rhai scene config ----------------

#[derive(Default, Clone)]
struct SceneCfg {
    atoms: Vec<(f64, f64, f64, bool)>,      // x, y, phi_deg, pinned
    rings: Vec<Ring2d>,
    arena: Option<(f64, f64)>,
    grid_dx: Option<f64>,
    w: Option<f64>, a: Option<f64>, b: Option<f64>,
    field: Option<bool>,
}

/// Run a .rhai scene script; registered fns fill SceneCfg.
fn load_scene(path: &std::path::Path) -> SceneCfg {
    let cfg = Rc::new(RefCell::new(SceneCfg::default()));
    let mut engine = rhai::Engine::new();
    {
        let c = cfg.clone();
        engine.register_fn("atom", move |x: f64, y: f64, phi: f64| c.borrow_mut().atoms.push((x, y, phi, false)));
    }
    {
        let c = cfg.clone();
        engine.register_fn("pinned", move |x: f64, y: f64, phi: f64| c.borrow_mut().atoms.push((x, y, phi, true)));
    }
    {
        let c = cfg.clone();
        engine.register_fn("random", move |n: i64, seed: i64, r: f64| {
            let mut r2 = Rng((seed as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1);
            for _ in 0..n {
                let rr = r * r2.next().sqrt();
                let t = r2.next() * std::f64::consts::TAU;
                let ph = r2.next() * 360.0;
                c.borrow_mut().atoms.push((rr * t.cos(), rr * t.sin(), ph, false));
            }
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("ring", move |x: f64, y: f64, nf: i64, r: f64| {
            let zc = Vec2d::new(0.0, 1.0);  // sites at 90°+k*60° (matches naphthalene scene)
            c.borrow_mut().rings.push(Ring2d { pos: Vec2d::new(x, y), zc, nfold: nf as u32, radius: r, a: 2.0, h: 0.5, a_core: 1.5, r0_core: r - 0.6, b_core: 2.5 });
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("ring_full", move |x: f64, y: f64, nf: i64, r: f64, a: f64, h: f64, ac: f64, rc: f64, bc: f64| {
            c.borrow_mut().rings.push(Ring2d { pos: Vec2d::new(x, y), zc: Vec2d::new(0.0, 1.0), nfold: nf as u32, radius: r, a, h, a_core: ac, r0_core: rc, b_core: bc });
        });
    }
    {
        let c = cfg.clone();
        engine.register_fn("arena", move |r: f64, k: f64| c.borrow_mut().arena = Some((r, k)));
    }
    {
        let c = cfg.clone();
        engine.register_fn("grid", move |dx: f64| c.borrow_mut().grid_dx = Some(dx));
    }
    {
        let c = cfg.clone();
        engine.register_fn("w", move |v: f64| c.borrow_mut().w = Some(v));
    }
    {
        let c = cfg.clone();
        engine.register_fn("a", move |v: f64| c.borrow_mut().a = Some(v));
    }
    {
        let c = cfg.clone();
        engine.register_fn("b", move |v: f64| c.borrow_mut().b = Some(v));
    }
    {
        let c = cfg.clone();
        engine.register_fn("field", move |v: bool| c.borrow_mut().field = Some(v));
    }
    match engine.eval_file::<rhai::Dynamic>(path.to_path_buf()) {
        Ok(_) => {}
        Err(e) => panic!("rhai scene {} failed: {e}", path.display()),
    }
    let out = cfg.borrow().clone();
    out
}

/// Build Rarff2d from a scene cfg. Pinned atoms get inv_m=inv_i=0.
fn build_ff(cfg: &SceneCfg) -> Rarff2d {
    let n = cfg.atoms.len();
    let types = vec![{ let mut t = TYPE_SP2;
        if let Some(w) = cfg.w { t.w = w; } if let Some(a) = cfg.a { t.a = a; } if let Some(b) = cfg.b { t.b = b; } t
    }; n];
    let pos: Vec<Vec2d> = cfg.atoms.iter().map(|a| Vec2d::new(a.0, a.1)).collect();
    let mut ff = Rarff2d::new(types, pos);
    for (i, a) in cfg.atoms.iter().enumerate() {
        ff.set_orient(i, a.2.to_radians());
        if a.3 { ff.inv_m[i] = 0.0; ff.inv_i[i] = 0.0; }   // pinned: no translation, no rotation
    }
    ff.set_rings(cfg.rings.clone());
    ff.arena = cfg.arena;
    if let Some(dx) = cfg.grid_dx {
        let r = cfg.arena.map(|a| a.0 + 2.0).unwrap_or(FIELD_EXT + 1.0);
        let ncell = (2.0 * r / dx).ceil() as usize + 1;
        ff.set_grid(dx, [-r, -r], ncell, ncell);
    }
    ff
}

struct App {
    window: Arc<Window>,
    ff: Rarff2d, rng: Rng, etot: f64,
    scene_path: PathBuf, scene_files: Vec<PathBuf>, eval_us: f64,
    run: bool, dt: f64, cdamp: f64, per_frame: usize,
    use_grid: bool, grid_dx: f64,
    pw: f64, pa: f64, pb: f64, pex_m: u32, ring_a: f64, ring_h: f64,
    selected: Option<usize>, dragging: Option<usize>,
    show_field: bool, show_bonds: bool, show_ports: bool, show_rings: bool, show_grid: bool,
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
        let cfg = load_scene(&scene_path);
        let ff = build_ff(&cfg);
        let mut app = Self {
            window, ff, rng: Rng(0x243F6A8885A308D3), etot: 0.0,
            scene_path, scene_files, eval_us: 0.0,
            run: true, dt: 0.02, cdamp: 0.95, per_frame: 40,
            use_grid: cfg.grid_dx.is_some(), grid_dx: cfg.grid_dx.unwrap_or(1.0),
            pw: cfg.w.unwrap_or(0.3), pa: cfg.a.unwrap_or(1.0), pb: cfg.b.unwrap_or(0.9), pex_m: molff::rarff2d::PEXP_M,
            ring_a: 2.0, ring_h: 0.5,
            selected: None, dragging: None,
            show_field: cfg.field.unwrap_or(true), show_bonds: true, show_ports: true, show_rings: true, show_grid: true,
            cam: TrackballCam::new(Vec3::new(0.0, 0.0, 0.0), 11.0),
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
        let mut v: Vec<PathBuf> = std::fs::read_dir(&dir).map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.extension().map(|e| e == "rhai").unwrap_or(false)).collect()).unwrap_or_default();
        v.sort();
        v
    }

    fn reload_scene(&mut self) {
        let cfg = load_scene(&self.scene_path);
        self.ff = build_ff(&cfg);
        self.use_grid = cfg.grid_dx.is_some();
        self.grid_dx = cfg.grid_dx.unwrap_or(self.grid_dx);
        self.show_field = cfg.field.unwrap_or(self.show_field);
        self.pw = cfg.w.unwrap_or(self.pw); self.pa = cfg.a.unwrap_or(self.pa); self.pb = cfg.b.unwrap_or(self.pb);
        self.dragging = None; self.selected = None;
        self.eval_once(); self.field_dirty = true;
        println!("reloaded scene {:?}: {} atoms, {} rings", self.scene_path.file_name(), self.ff.natom(), self.ff.rings.len());
    }

    fn apply_params(&mut self) {
        for t in &mut self.ff.types { t.w = self.pw; t.a = self.pa; t.b = self.pb; }
        let rc_old = self.ff.rcut;
        self.ff.set_pex_m(self.pex_m);   // recompute rcut always (b slider changes support too)
        if (self.ff.rcut - rc_old).abs() > 1e-12 { self.ff.grid = None; }
        for r in &mut self.ff.rings { r.a = self.ring_a; r.h = self.ring_h; }
        if self.use_grid {
            if self.ff.grid.is_none() || (self.ff.grid.as_ref().unwrap().dx - self.grid_dx).abs() > 1e-9 {
                let r = self.ff.arena.map(|a| a.0 + 2.0).unwrap_or(FIELD_EXT + 1.0);
                let ncell = (2.0 * r / self.grid_dx).ceil() as usize + 1;
                self.ff.set_grid(self.grid_dx, [-r, -r], ncell, ncell);
            }
        } else { self.ff.grid = None; }
        self.eval_once();   // refresh forces + cell_of so grid viz/eatom aren't stale while paused
        self.field_dirty = true;
    }

    fn eval_once(&mut self) {
        let t = std::time::Instant::now();
        self.etot = self.ff.eval();
        self.eval_us = self.eval_us * 0.9 + t.elapsed().as_secs_f64() * 1e6 * 0.1;
    }

    fn mouse_z0(&self, m: Vec2) -> Option<Vec2d> {
        let (ro, rd) = self.cam.screen_ray(m, self.window_size.0, self.window_size.1);
        if rd.z.abs() < 1e-6 { return None; }
        let t = -ro.z / rd.z;
        if t < 0.0 { return None; }
        let p = ro + rd * t;
        Some(Vec2d::new(p.x as f64, p.y as f64))
    }

    fn pick_atom(&self, m: Vec2) -> Option<usize> {
        let p = self.mouse_z0(m)?;
        let mut best: Option<(usize, f64)> = None;
        for i in 0..self.ff.natom() {
            let d = (self.ff.pos[i] - p).norm();
            if d < PICK_R && best.map(|(_, db)| d < db).unwrap_or(true) { best = Some((i, d)); }
        }
        best.map(|(i, _)| i)
    }

    fn step(&mut self) {
        for _ in 0..self.per_frame {
            if let Some(i) = self.dragging {
                if let Some(p) = self.mouse_z0(self.mouse_now) { self.ff.pos[i] = p; self.ff.vel[i].set(0.0, 0.0); }
            }
            let t = std::time::Instant::now();
            self.ff.eval();
            self.eval_us = self.eval_us * 0.95 + t.elapsed().as_secs_f64() * 1e6 * 0.05;
            self.ff.step_md(self.dt, self.cdamp);
        }
        self.eval_once();
        self.field_dirty = true;
    }

    fn update_field(&mut self) {
        if !self.field_dirty { return; }
        self.field_dirty = false;
        let skip = self.dragging.or(self.selected).unwrap_or(usize::MAX);
        let n = FIELD_N as usize;
        let mut buf = vec![0u8; n * n * 4];
        for iy in 0..n {
            let y = -FIELD_EXT + 2.0 * FIELD_EXT * (iy as f64 + 0.5) / n as f64;
            for ix in 0..n {
                let x = -FIELD_EXT + 2.0 * FIELD_EXT * (ix as f64 + 0.5) / n as f64;
                let c = field_color(self.ff.field_at(Vec2d::new(x, y), skip, Rarff2dType { w: self.pw, a: self.pa, b: self.pb, ..TYPE_SP2 }));
                buf[(iy * n + ix) * 4..(iy * n + ix) * 4 + 4].copy_from_slice(&c);
            }
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.field_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &buf, wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * FIELD_N), rows_per_image: Some(FIELD_N) },
            wgpu::Extent3d { width: FIELD_N, height: FIELD_N, depth_or_array_layers: 1 });
    }

    fn collect_lines(&self) -> Vec<LineVertex> {
        let mut lines = Vec::new();
        let lv = |x: f32, y: f32, z: f32, c: [f32; 4]| LineVertex { pos: [x, y, z], col: c };
        if self.show_bonds {
            for i in 0..self.ff.natom() { for j in (i + 1)..self.ff.natom() {
                let d = self.ff.pos[j] - self.ff.pos[i];
                let r = d.norm();
                if r < BOND_R && r > 1e-9 {
                    let th = d.y.atan2(d.x);
                    let y = gate(3, self.pw, th - self.ff.phi(i)).0 * gate(3, self.pw, th + std::f64::consts::PI - self.ff.phi(j)).0;
                    if y > GATE_Y_MIN {
                        let a = 0.4 + 0.6 * y as f32;
                        lines.push(lv(self.ff.pos[i].x as f32, self.ff.pos[i].y as f32, 0.02, [a, a, a, 1.0]));
                        lines.push(lv(self.ff.pos[j].x as f32, self.ff.pos[j].y as f32, 0.02, [a, a, a, 1.0]));
                    }
                }
            }}
        }
        if self.show_ports {
            for i in 0..self.ff.natom() {
                for k in 0..3u32 {
                    let ph = self.ff.phi(i) + k as f64 * std::f64::consts::TAU / 3.0;
                    let (dx, dy) = (0.42 * ph.cos(), 0.42 * ph.sin());
                    lines.push(lv(self.ff.pos[i].x as f32, self.ff.pos[i].y as f32, 0.05, [0.2, 1.0, 0.4, 0.9]));
                    lines.push(lv((self.ff.pos[i].x + dx) as f32, (self.ff.pos[i].y + dy) as f32, 0.05, [0.2, 1.0, 0.4, 0.9]));
                }
                if self.ff.inv_m[i] == 0.0 {   // pinned marker: dark cross
                    let (x, y) = (self.ff.pos[i].x as f32, self.ff.pos[i].y as f32);
                    lines.push(lv(x - 0.15, y, 0.06, [0.1, 0.1, 0.1, 1.0])); lines.push(lv(x + 0.15, y, 0.06, [0.1, 0.1, 0.1, 1.0]));
                    lines.push(lv(x, y - 0.15, 0.06, [0.1, 0.1, 0.1, 1.0])); lines.push(lv(x, y + 0.15, 0.06, [0.1, 0.1, 0.1, 1.0]));
                }
            }
        }
        if self.show_rings {
            for rg in &self.ff.rings {
                let ns = 64;
                let phi_c = rg.zc.y.atan2(rg.zc.x);
                for s in 0..ns {
                    let (t0, t1) = (s as f64 * std::f64::consts::TAU / ns as f64, (s + 1) as f64 * std::f64::consts::TAU / ns as f64);
                    lines.push(lv((rg.pos.x + rg.radius * t0.cos()) as f32, (rg.pos.y + rg.radius * t0.sin()) as f32, 0.05, [0.1, 0.8, 0.3, 0.7]));
                    lines.push(lv((rg.pos.x + rg.radius * t1.cos()) as f32, (rg.pos.y + rg.radius * t1.sin()) as f32, 0.05, [0.1, 0.8, 0.3, 0.7]));
                }
                for k in 0..rg.nfold {
                    let t = phi_c + k as f64 * std::f64::consts::TAU / rg.nfold as f64;
                    lines.push(lv((rg.pos.x + (rg.radius - 0.12) * t.cos()) as f32, (rg.pos.y + (rg.radius - 0.12) * t.sin()) as f32, 0.05, [0.4, 1.0, 0.5, 0.9]));
                    lines.push(lv((rg.pos.x + (rg.radius + 0.12) * t.cos()) as f32, (rg.pos.y + (rg.radius + 0.12) * t.sin()) as f32, 0.05, [0.4, 1.0, 0.5, 0.9]));
                }
            }
            if let Some((ra, _)) = self.ff.arena {
                let ns = 96;
                for s in 0..ns {
                    let (t0, t1) = (s as f64 * std::f64::consts::TAU / ns as f64, (s + 1) as f64 * std::f64::consts::TAU / ns as f64);
                    lines.push(lv((ra * t0.cos()) as f32, (ra * t0.sin()) as f32, 0.05, [0.4, 0.4, 0.4, 0.6]));
                    lines.push(lv((ra * t1.cos()) as f32, (ra * t1.sin()) as f32, 0.05, [0.4, 0.4, 0.4, 0.6]));
                }
            }
        }
        // --- grid cells (neighbor-search visualization) ---
        if self.show_grid {
            if let Some(g) = &self.ff.grid {
                let (ox, oy) = (g.origin[0] as f32, g.origin[1] as f32);
                let dx = g.dx as f32;
                let rect = |lines: &mut Vec<LineVertex>, cx: usize, cy: usize, col: [f32; 4], z: f32| {
                    let (x0, y0) = (ox + cx as f32 * dx, oy + cy as f32 * dx);
                    let (x1, y1) = (x0 + dx, y0 + dx);
                    lines.push(lv(x0, y0, z, col)); lines.push(lv(x1, y0, z, col));
                    lines.push(lv(x1, y0, z, col)); lines.push(lv(x1, y1, z, col));
                    lines.push(lv(x1, y1, z, col)); lines.push(lv(x0, y1, z, col));
                    lines.push(lv(x0, y1, z, col)); lines.push(lv(x0, y0, z, col));
                };
                for cy in 0..g.ny { for cx in 0..g.nx {                                     // all domain cells (faint)
                    rect(&mut lines, cx, cy, [0.35, 0.35, 0.45, 0.35], -0.03);
                }}
                for c in g.cell_of.iter().collect::<std::collections::HashSet<_>>() {       // occupied cells
                    rect(&mut lines, (c % g.nx as i32) as usize, (c / g.nx as i32) as usize, [0.2, 0.5, 0.9, 0.8], -0.02);
                }
                if let Some(i) = self.selected.or(self.dragging) {                          // stencil of selected atom
                    let c = g.cell_of[i] as i64;
                    let s = (self.ff.rcut / g.dx).ceil() as i64;
                    let (nx, ny) = (g.nx as i64, g.ny as i64);
                    for cy in (c / nx - s).max(0)..=(c / nx + s).min(ny - 1) {
                        for cx in (c % nx - s).max(0)..=(c % nx + s).min(nx - 1) {
                            rect(&mut lines, cx as usize, cy as usize, [0.9, 0.7, 0.1, 0.6], -0.02);
                        }
                    }
                }
            }
        }
        // --- cutoff circles around selected/dragged atom ---
        if let Some(i) = self.selected.or(self.dragging) {
            let (ax, ay) = (self.ff.pos[i].x as f32, self.ff.pos[i].y as f32);
            let ns = 96;
            // rcut (pair-search cutoff) — cyan
            for s in 0..ns {
                let (t0, t1) = (s as f64 * std::f64::consts::TAU / ns as f64, (s + 1) as f64 * std::f64::consts::TAU / ns as f64);
                let r = self.ff.rcut as f32;
                lines.push(lv(ax + r * t0.cos() as f32, ay + r * t0.sin() as f32, 0.04, [0.0, 0.8, 1.0, 0.7]));
                lines.push(lv(ax + r * t1.cos() as f32, ay + r * t1.sin() as f32, 0.04, [0.0, 0.8, 1.0, 0.7]));
            }
            // pex support r_sup = r0 + 2^m/b = rcut (auto-derived, cyan circle
            // already shows it) — draw dashed magenta to mark they coincide
            let r_sup = self.ff.rcut as f32;
            for s in 0..ns {
                let (t0, t1) = (s as f64 * std::f64::consts::TAU / ns as f64, (s + 1) as f64 * std::f64::consts::TAU / ns as f64);
                lines.push(lv(ax + r_sup * t0.cos() as f32, ay + r_sup * t0.sin() as f32, 0.04, [1.0, 0.2, 1.0, 0.5]));
                lines.push(lv(ax + r_sup * t1.cos() as f32, ay + r_sup * t1.sin() as f32, 0.04, [1.0, 0.2, 1.0, 0.5]));
            }
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
                                if self.ff.inv_m[i] > 0.0 { self.dragging = Some(i); }  // pinned atoms can't be dragged
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
                    "e" | "E" => { self.show_rings = !self.show_rings; }
                    "g" | "G" => { self.show_grid = !self.show_grid; }
                    "r" | "R" => { self.scene_files = Self::list_scenes(); self.reload_scene(); }
                    "a" | "A" => {
                        if let Some(p) = self.mouse_z0(self.mouse_now) {
                            self.ff.pos.push(p); self.ff.types.push(Rarff2dType { w: self.pw, a: self.pa, b: self.pb, ..TYPE_SP2 });
                            self.ff.zc.push(Vec2d::new(1.0, 0.0));
                            self.ff.set_orient(self.ff.natom() - 1, self.rng.next() * std::f64::consts::TAU);
                            self.ff.vel.push(Vec2d::new(0.0, 0.0)); self.ff.om.push(0.0);
                            self.ff.force.push(Vec2d::new(0.0, 0.0)); self.ff.torq.push(0.0);
                            self.ff.eatom.push(0.0); self.ff.inv_m.push(1.0); self.ff.inv_i.push(1.0);
                            if self.use_grid { self.ff.grid = None; self.apply_params(); }  // natom changed -> rebuild grid
                            self.eval_once(); self.field_dirty = true;
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn prepare(&mut self) {
        if self.run { self.step(); }
        self.instances = (0..self.ff.natom()).map(|i| AtomInstance {
            pos: [self.ff.pos[i].x as f32, self.ff.pos[i].y as f32, 0.0], radius: 0.32,
            color: if self.dragging == Some(i) { [1.0, 0.5, 0.9] }
                   else if self.ff.inv_m[i] == 0.0 { [0.15, 0.15, 0.15] }       // pinned = dark
                   else if self.selected == Some(i) { [1.0, 0.8, 0.2] }
                   else { eatom_color(self.ff.eatom[i]) }, _pad: 0.0,
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
            self.surface_renderer.render(&mut encoder, &view, self.renderer.depth_view(), &cam, &tex_view,
                [-FIELD_EXT as f32, -FIELD_EXT as f32, FIELD_Z], [2.0 * FIELD_EXT as f32, 0.0, 0.0], [0.0, 2.0 * FIELD_EXT as f32, 0.0]);
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
        egui::Window::new("rarff2d").anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
            .frame(egui::Frame::none().fill(egui::Color32::from_rgba_premultiplied(0, 0, 0, 190)))
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(format!("scene: {}", self.scene_path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default())).size(15.0).color(egui::Color32::WHITE));
                let mut pick: Option<usize> = None;
                ui.horizontal(|ui| {
                    for (i, p) in self.scene_files.iter().enumerate() {
                        if ui.small_button(p.file_stem().map(|s| s.to_string_lossy().replace("rarff2d_", "")).unwrap_or_default()).clicked() { pick = Some(i); }
                    }
                    if ui.small_button("reload [R]").clicked() { pick = Some(usize::MAX); }
                });
                match pick {
                    Some(i) if i < self.scene_files.len() => { self.scene_path = self.scene_files[i].clone(); self.reload_scene(); }
                    Some(_) => { self.scene_files = Self::list_scenes(); self.reload_scene(); }
                    None => {}
                }
                ui.label(egui::RichText::new(format!("E_total = {:9.4} eV   eval = {:.1} us/step", self.etot, self.eval_us)).size(13.0));
                ui.label(egui::RichText::new(format!("rcut = r0+2^{}/b = {:.2} A (auto from pex_m, cyan=magenta)", self.ff.pex_m, self.ff.rcut)).size(11.0));
                if let Some(i) = self.selected { ui.label(egui::RichText::new(format!("atom {i}: E={:.3} phi={:.0}deg{}", self.ff.eatom[i], self.ff.phi(i).to_degrees(), if self.ff.inv_m[i] == 0.0 { " [pinned]" } else { "" })).size(12.0).color(egui::Color32::YELLOW)); }
                ui.separator();
                ui.label("pair potential");
                params_changed |= ui.add(egui::Slider::new(&mut self.pw, 0.1..=1.2).text("w (gate width)")).changed();
                params_changed |= ui.add(egui::Slider::new(&mut self.pa, 0.0..=3.0).text("a (depth)")).changed();
                params_changed |= ui.add(egui::Slider::new(&mut self.pb, 0.3..=3.0).text("b (steepness)")).changed();
                params_changed |= ui.add(egui::Slider::new(&mut self.pex_m, 1..=6).text("pex m (cutoff: r0+2^m/b)")).changed();
                if !self.ff.rings.is_empty() {
                    params_changed |= ui.add(egui::Slider::new(&mut self.ring_a, 0.0..=4.0).text("ring a")).changed();
                    params_changed |= ui.add(egui::Slider::new(&mut self.ring_h, 0.1..=1.2).text("ring h")).changed();
                }
                ui.separator();
                ui.label("solver / neighbors");
                ui.add(egui::Slider::new(&mut self.dt, 0.005..=0.05).text("dt"));
                ui.add(egui::Slider::new(&mut self.cdamp, 0.7..=0.999).text("damping"));
                ui.add(egui::Slider::new(&mut self.per_frame, 1..=400).text("steps/frame"));
                params_changed |= ui.checkbox(&mut self.use_grid, "use CSR grid").changed();
                if self.use_grid { params_changed |= ui.add(egui::Slider::new(&mut self.grid_dx, 0.5..=4.0).text("grid dx [A]")).changed(); }
                ui.separator();
                ui.label(egui::RichText::new("LMB drag · A add · N/A keys · Space run\nF field B bonds T ports E rings G cells").size(10.5).color(egui::Color32::GRAY));
            });
        if params_changed { self.apply_params(); }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../../..");
    // --scene accepts a bare filename (resolved in data/scenes/) or a path
    let scene = args.windows(2).find(|w| w[0] == "--scene").map(|w| {
        let p = PathBuf::from(&w[1]);
        if p.is_file() { p } else { let q = root.join(SCENE_DIR).join(&p); assert!(q.is_file(), "scene not found: {} (tried {})", p.display(), q.display()); q }
    }).unwrap_or_else(|| root.join(SCENE_DIR).join("rarff2d_pair_probe.rhai"));
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(event_loop.create_window(Window::default_attributes().with_title(format!("rarff2d_view — {}", scene.display())).with_inner_size(winit::dpi::PhysicalSize::new(1400, 900))).unwrap());
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
