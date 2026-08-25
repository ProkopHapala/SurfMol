use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use glam::{Quat, Vec2, Vec3};
use surfmol::mol_world::{BondedFFMode, MolWorld};
use molff::nonbonded::NonBondedFF;
use molrender::impostor::{AtomInstance, ImpostorRenderer};
use molrender::line_renderer::{LineRenderer, LineVertex};
use molrender::surface_renderer::SurfaceRenderer;
use numtypes::Vec3d;
use moltopo::xyz;
use moltopo::assign_uff;
use moltopo::builder;
use moltopo::params::{Params, get_reqh};
use molgui::gui::trackball::TrackballCam;
use molgui::gui::kekule_editor::{KekuleEditor, EditMode, AtomType, collect_hex_grid_points, collect_builder_bonds, collect_builder_atoms, export_xyz, builder_summary, element_color};
use molgui::gui::clipboard::{Clipboard, inject_cut_copy_if_needed, inject_paste_if_needed, handle_output_commands};
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes};

const ATOM_SCALE: f32 = 0.25;
const K_PICK: f64 = 30.0;
const PER_FRAME: i32 = 100;
const PICK_RAY_R: f32 = 0.5;
const LATTICE_A: f64 = 5.66;  // NaCl conventional cell, Na-Cl = a/2 = 2.83 Å
const SURFACE_Z0: f32 = 0.0;
const BETA_CHARGE: f64 = 0.3;      // electrostatics z-decay (slower)
const BETA_MORSE_RATIO: f64 = 2.0; // Morse decay = ratio * charge decay (steeper vdW)
const Q_AMP: f64 = 1.0;
const PLQ_AMP: f64 = 1.0;
const SURFACE_GRID_N: i32 = 256;
const SURFACE_SIZE: f32 = 10.0;
const GROUP_SIZE_DEFAULT: usize = 32;

fn ray_sphere(ro: Vec3, rd: Vec3, sc: Vec3, sr: f32) -> Option<f32> {
    let oc = ro - sc; let b = oc.dot(rd); let c = oc.dot(oc) - sr * sr; let disc = b * b - c;
    if disc < 0.0 { return None; } let t = -b - disc.sqrt(); if t >= 0.0 { Some(t) } else { None }
}

fn get_force_spring_ray(p: Vec3, hray: Vec3, ray0: Vec3, k: f32) -> Vec3 {
    let dp = p - ray0; let cdot = hray.dot(dp); let dp_perp = dp - hray * cdot; -dp_perp * k
}

#[derive(Default)]
struct Dirty { atoms: bool, camera: bool, bonds: bool, surface: bool, groups: bool }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LabelMode { None, AtomNumber, AtomType, Charge, ElementName }

struct App {
    window: Arc<winit::window::Window>,
    instance: Arc<wgpu::Instance>,
    world: MolWorld, elems: Vec<String>, params: Params, uff_types: Vec<String>, charges: Vec<f64>,
    cam: TrackballCam, selected: Option<usize>, pinned: Vec<bool>, pick_k: f64,
    show_bonds: bool, show_surface: bool, show_help: bool, show_groups: bool, show_ports: bool, show_labels: bool, show_debug_cursor: bool, label_mode: LabelMode,
    run_relax: bool, dt: f64, flim: f64, damping: f64, zero_v_on_opposition: bool, per_frame: i32,
    dirty: Dirty,
    device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: wgpu::SurfaceConfiguration,
    renderer: ImpostorRenderer, instances: Vec<AtomInstance>, line_renderer: LineRenderer, surface_renderer: SurfaceRenderer,
    surface_texture: Option<wgpu::Texture>, surface_origin: [f32; 3], surface_u: [f32; 3], surface_v: [f32; 3],
    mouse_now: Vec2, mouse_delta: Vec2, prev_mouse: Vec2, lmb_down: bool, mouse_down: Vec2,
    trackballing: bool, trackball_prev: Vec2, window_size: (f32, f32),
    surface: wgpu::Surface<'static>,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    clipboard: Clipboard,
    etot: f64, eb: f64, ea: f64, ed: f64, ei: f64, enb: f64, es: f64,
    // --- Kekule editor state ---
    kekule_editor: KekuleEditor,
    builder: builder::Builder,
    show_kekule_editor: bool,
    show_hex_grid: bool,
    show_ghost_hexes: bool,
    edit_from_builder: bool,
}

impl App {
    fn new(window: Arc<winit::window::Window>) -> Self {
        let instance = Arc::new(wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        }));
        let surface = instance.create_surface(window.clone()).expect("create_surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("no adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            label: Some("apps_device"),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        })).expect("no device");
        let device = Arc::new(device); let queue = Arc::new(queue);
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|&f| f == wgpu::TextureFormat::Rgba8UnormSrgb).or_else(|| caps.formats.iter().copied().find(|f| f.is_srgb())).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration { usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format, width: size.width.max(1), height: size.height.max(1), present_mode: caps.present_modes[0], alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2 };
        surface.configure(&device, &config);
        let (ww, wh) = (size.width as f32, size.height as f32);

        let egui_ctx = egui::Context::default();
        let viewport_id = egui_ctx.viewport_id();
        let egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, Some(window.scale_factor() as f32), None, None);
        let egui_renderer = egui_wgpu::Renderer::new(&*device, config.format, egui_wgpu::RendererOptions::default());
        let line_renderer = LineRenderer::new(device.clone(), queue.clone(), config.format);
        let surface_renderer = SurfaceRenderer::new(device.clone(), queue.clone(), config.format);

        // --- World setup (ported from old main.rs) ---
        let workspace_root = std::path::PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../../..");
        let args: Vec<String> = std::env::args().collect();
        let mut copies_x: usize = 1; let mut copies_y: usize = 1; let mut spacing: f64 = 12.0;
        let mut group_size: usize = GROUP_SIZE_DEFAULT; let mut per_frame: i32 = PER_FRAME; let mut dt: f64 = 0.02;
        { let mut it = args.iter().skip(1); while let Some(a) = it.next() { match a.as_str() { "--copies-x" => copies_x = it.next().unwrap_or(&"1".to_string()).parse().unwrap_or(1), "--copies-y" => copies_y = it.next().unwrap_or(&"1".to_string()).parse().unwrap_or(1), "--spacing" => spacing = it.next().unwrap_or(&"12.0".to_string()).parse().unwrap_or(12.0), "--group-size" => group_size = it.next().unwrap_or(&"32".to_string()).parse().unwrap_or(GROUP_SIZE_DEFAULT), "--perFrame" => per_frame = it.next().unwrap_or(&"100".to_string()).parse().unwrap_or(PER_FRAME), "--dt" => dt = it.next().unwrap_or(&"0.02".to_string()).parse().unwrap_or(0.02), _ => {} } } }
        let xyz_path: PathBuf = args.iter().skip(1).find(|s| !s.starts_with("--")).map(|s| { let p = PathBuf::from(s); if p.is_absolute() { p } else { workspace_root.join(p) } }).unwrap_or_else(|| workspace_root.join("data/xyz/pentacene.xyz"));
        println!("Loading XYZ: {:?}", xyz_path);
        let sys = xyz::read_xyz(&xyz_path).expect("read_xyz failed"); println!("Loaded {} atoms", sys.elems.len());

        let mut apos_v3d = Vec::<Vec3d>::new(); let mut elems = Vec::<String>::new(); let mut charges = Vec::<f64>::new();
        for iy in 0..copies_y { for ix in 0..copies_x { let shift = Vec3d::new((ix as f64) * spacing, (iy as f64) * spacing, 0.0); let i0 = apos_v3d.len(); apos_v3d.extend(sys.apos.iter().map(|p| Vec3d::set_add(*p, shift))); elems.extend(sys.elems.iter().cloned()); charges.extend(sys.charges.iter().copied()); assert!(apos_v3d.len() - i0 == sys.apos.len()); } }
        println!("Spawned copies: {}x{} -> natoms={}", copies_x, copies_y, apos_v3d.len());

        let dat_dir_candidates = [workspace_root.join("tmp/FireCore_cpp/common_resources"), workspace_root.join("data")];
        let dat_dir = dat_dir_candidates.iter().find(|d| d.join("ElementTypes.dat").exists() && d.join("AtomTypes.dat").exists() && d.join("BondTypes.dat").exists() && d.join("AngleTypes.dat").exists()).cloned().unwrap_or_else(|| dat_dir_candidates[0].clone());
        let mut params = Params::new();
        let have_params = dat_dir.join("ElementTypes.dat").exists() && dat_dir.join("AtomTypes.dat").exists() && dat_dir.join("BondTypes.dat").exists() && dat_dir.join("AngleTypes.dat").exists();
        if have_params { params.load_element_types(dat_dir.join("ElementTypes.dat")); params.load_atom_types(dat_dir.join("AtomTypes.dat")); params.load_bond_types(dat_dir.join("BondTypes.dat")); params.load_angle_types(dat_dir.join("AngleTypes.dat")); if dat_dir.join("DihedralTypes.dat").exists() { params.load_dihedral_types(dat_dir.join("DihedralTypes.dat")); } println!("Loaded {} elements, {} atom types, {} bond types", params.elements.len(), params.atom_types.len(), params.bonds.len()); }
        else { println!("WARNING: .dat files not found in {:?}; running with dummy radii/REQs/bond params", dat_dir); }

        let radii: Vec<f64> = if have_params { elems.iter().map(|el| params.get_element_type(el).map(|et| et.r_cov).unwrap_or(1.0)).collect() } else { elems.iter().map(|el| match el.as_str() { "H" => 0.31, "C" => 0.76, "N" => 0.71, "O" => 0.66, _ => 1.0 }).collect() };
        let mut b = builder::Builder::from_positions_and_radii(&apos_v3d, &elems, &radii, 0.4);
        let top = b.bake(); let mut world = MolWorld::from_topology(&top);
        world.make_neigh_bs(); world.bake_angle_neighs(); world.bake_dihedral_neighs(); world.bake_inversion_neighs(); world.map_atom_interactions();
        world.nonbonded = Some(NonBondedFF::new(world.natoms()));
        let natoms = world.natoms();
        let neighs = world.dyn_atoms.atoms.neighs.as_slice().to_vec();
        world.nonbonded.as_mut().unwrap().make_second_neighs(&neighs, natoms);
        world.nonbonded.as_mut().unwrap().set_cutoff(8.0);

        let uff_types: Vec<String>;
        { let neighs: Vec<[i32; 4]> = world.dyn_atoms.atoms.neighs.as_slice().iter().map(|q| q.as_array()).collect(); uff_types = assign_uff::assign_uff_types(&elems, &neighs); let mut counts: HashMap<String, usize> = HashMap::new(); for t in &uff_types { *counts.entry(t.clone()).or_insert(0) += 1; } let mut kv: Vec<(String, usize)> = counts.into_iter().collect(); kv.sort_unstable_by(|a, b| b.1.cmp(&a.1)); println!("=== UFF type histogram ==="); for (t, c) in kv.iter() { println!("{:6}  {}", t, c); }
        let has_sp2 = uff_types.iter().any(|t| matches!(t.as_str(), "C_R"|"C_2"|"N_R"|"O_2"|"O_R")); if has_sp2 { world.bonded_mode = BondedFFMode::Uff; println!("Detected sp2/aromatic types -> default bonded_mode = Uff"); }
        world.rigid_sp3.set_port_geometry_from_types(&uff_types);
        if have_params { for i in 0..world.natoms() { let t = uff_types[i].as_str(); let mut req = get_reqh(&params, t); if charges[i] != 0.0 { req[2] = charges[i]; } world.nonbonded.as_mut().unwrap().reqs.as_mut_slice()[i] = req; }
        world.nonbonded.as_mut().unwrap().make_plqs(2.0); println!("=== Atom types + charges ==="); for i in 0..world.natoms() { let q = world.nonbonded.as_ref().unwrap().reqs.as_slice()[i][2]; println!("atom {:3} el {:2} type {:6} Q {:8.4}", i, elems[i], uff_types[i], q); }
        for ib in 0..world.uff.nbonds as usize { let b = world.uff.bon_atoms.as_slice()[ib]; let ia = b[0] as usize; let ja = b[1] as usize; let a = elems[ia].as_str(); let b = elems[ja].as_str(); if let Some(bp) = params.get_bond_param(a, b, 1) { world.uff.bon_params.as_mut_slice()[ib] = [bp.k, bp.l0]; } else { panic!("missing bond param for {}-{} order=1", a, b); } }
        for ia in 0..world.uff.nangles as usize { let ang = world.uff.ang_atoms.as_slice()[ia]; let i0 = ang[0] as usize; let i1 = ang[1] as usize; let i2 = ang[2] as usize; let a = elems[i0].as_str(); let b = elems[i1].as_str(); let c = elems[i2].as_str(); let ap = params.get_angle_param(a, b, c).unwrap_or_else(|| panic!("missing angle param for {}-{}-{}", a, b, c)); let th0 = ap.a0.to_radians(); let ct = th0.cos(); let st2 = 1.0 - ct * ct; assert!(st2 > 1e-12, "invalid angle theta0={} deg leads to sin^2(theta0)~0", ap.a0); let c2 = 1.0 / (4.0 * st2); let c1 = -4.0 * c2 * ct; let c0 = c2 * (2.0 * ct * ct + 1.0); world.uff.ang_params.as_mut_slice()[ia] = [ap.k, c0, c1, c2, 0.0]; }
        for id in 0..world.uff.ndihedrals as usize { let d = world.uff.dih_atoms.as_slice()[id]; let a = uff_types[d.x as usize].as_str(); let b = uff_types[d.y as usize].as_str(); let c = uff_types[d.z as usize].as_str(); let e = uff_types[d.w as usize].as_str(); if let Some(dp) = params.get_dihedral_param(a, b, c, e, 1) { let a0 = dp.a0.to_radians(); let n = dp.n as f64; let phase = n * a0; let s = phase.sin().abs(); if s > 1e-3 { panic!("dihedral phase not supported by current Uff dihedral form: {}-{}-{}-{} a0={}deg n={} => n*a0={}deg", a, b, c, e, dp.a0, dp.n, phase.to_degrees()); } let dsign = if phase.cos() >= 0.0 { 1.0 } else { -1.0 }; world.uff.dih_params.as_mut_slice()[id] = [dp.k, dsign, dp.n as f64]; } else { world.uff.dih_params.as_mut_slice()[id] = [0.0, 1.0, 3.0]; } }
        for ii in 0..world.uff.ninversions as usize { let inv = world.uff.inv_atoms.as_slice()[ii]; let ic = inv.x as usize; let t = uff_types[ic].as_str(); if matches!(t, "C_R"|"C_2"|"N_R"|"O_2"|"O_R") { world.uff.inv_params.as_mut_slice()[ii] = [50.0, 1.0, -1.0, 0.0]; } else { world.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0]; } }
        } else { for i in 0..world.natoms() { let mut req = [1.5, 0.1, 0.0, 0.0]; if charges[i] != 0.0 { req[2] = charges[i]; } world.nonbonded.as_mut().unwrap().reqs.as_mut_slice()[i] = req; } world.nonbonded.as_mut().unwrap().make_plqs(2.0); let apos_slice = world.dyn_atoms.atoms.apos.as_slice(); for ib in 0..world.uff.nbonds as usize { let b = world.uff.bon_atoms.as_slice()[ib]; let ia = b[0] as usize; let ja = b[1] as usize; let d = numtypes::Vec3d::set_sub(apos_slice[ja], apos_slice[ia]); let l0 = d.norm(); world.uff.bon_params.as_mut_slice()[ib] = [100.0, l0]; } }
        }
        for i in 0..world.natoms() { world.dyn_atoms.atoms.apos.as_mut_slice()[i].z += 2.0; }
        world.update_hneigh();
        world.setup_nacl_surface(LATTICE_A, SURFACE_Z0 as f64, BETA_CHARGE, BETA_MORSE_RATIO, Q_AMP, PLQ_AMP);
        println!("Surface setup complete (NaCl lattice a={} Å)", LATTICE_A);

        let mut cam = TrackballCam::new(Vec3::new(0.0, 1.0, 0.0), 6.0);

        if natoms > 0 {
            let ps = world.dyn_atoms.atoms.apos.as_slice();
            let mut mn = Vec3::new(ps[0].x as f32, ps[0].y as f32, ps[0].z as f32);
            let mut mx = mn;
            for i in 0..natoms {
                let p = Vec3::new(ps[i].x as f32, ps[i].y as f32, ps[i].z as f32);
                mn = mn.min(p);
                mx = mx.max(p);
            }
            let target = (mn + mx) * 0.5;
            let span = (mx - mn).max(Vec3::splat(2.0));
            let max_span = span.x.max(span.y).max(span.z);
            cam.target = target;
            cam.zoom = max_span * 1.6;
            cam.dist_cam = max_span * 2.5;
            cam.rotation = Quat::IDENTITY;
        }

        let ps = world.dyn_atoms.atoms.apos.as_slice();
        let mut instances = Vec::with_capacity(natoms);
        for i in 0..natoms {
            let el = elems.get(i).map(|s| s.as_str()).unwrap_or("C");
            let col = params.element_color_f32(el);
            let r = params.element_radius_vdw(el) * ATOM_SCALE;
            let p = &ps[i];
            instances.push(AtomInstance { pos: [p.x as f32, p.y as f32, p.z as f32], radius: r, color: col, _pad: 0.0 });
        }
        let mut renderer = ImpostorRenderer::new(device.clone(), queue.clone(), natoms.max(1), config.format);
        renderer.set_target_size(config.width, config.height);
        renderer.set_atoms(&instances);
        let mut dirty = Dirty::default(); dirty.atoms = false; dirty.camera = true; dirty.bonds = false; dirty.surface = false; dirty.groups = false;
        println!("App initialized. Controls: H=help  SPACE=relax  S=surface  B=bonds  P=pin  ESC=deselect");
        let kekule_editor = KekuleEditor::new();
        let mut app = Self { window, instance, world, elems, params, uff_types, charges, cam, selected: None, pinned: vec![false; natoms], pick_k: K_PICK, show_bonds: true, show_surface: true, show_help: true, show_groups: false, show_ports: false, show_labels: true, show_debug_cursor: true, label_mode: LabelMode::ElementName, run_relax: false, dt, flim: 1000.0, damping: 0.0, zero_v_on_opposition: true, per_frame, dirty, device, queue, config, renderer, instances, line_renderer, surface_renderer, surface_texture: None, surface_origin: [0.0; 3], surface_u: [0.0; 3], surface_v: [0.0; 3], mouse_now: Vec2::ZERO, mouse_delta: Vec2::ZERO, prev_mouse: Vec2::ZERO, lmb_down: false, mouse_down: Vec2::ZERO, trackballing: false, trackball_prev: Vec2::ZERO, window_size: (ww, wh), surface, egui_ctx, egui_state, egui_renderer, clipboard: Clipboard::new(), etot: 0.0, eb: 0.0, ea: 0.0, ed: 0.0, ei: 0.0, enb: 0.0, es: 0.0, kekule_editor, builder: b, show_kekule_editor: true, show_hex_grid: true, show_ghost_hexes: true, edit_from_builder: false };
        app.rebuild_surface_cache();
        app
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        let w = new_size.width.max(1);
        let h = new_size.height.max(1);
        if w == self.config.width && h == self.config.height { return; }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.renderer.set_target_size(w, h);
    }

    fn sync_pos_from_engine(&mut self) {
        self.dirty.atoms = true;
    }

    fn pick_atom(&self, mouse: Vec2) -> Option<usize> {
        let (ro, rd) = self.cam.screen_ray(mouse, self.window_size.0, self.window_size.1);
        let mut best_t = f32::MAX; let mut best_i = None;
        let ps = self.world.dyn_atoms.atoms.apos.as_slice();
        for i in 0..self.world.natoms() {
            let p = Vec3::new(ps[i].x as f32, ps[i].y as f32, ps[i].z as f32);
            if let Some(t) = ray_sphere(ro, rd, p, PICK_RAY_R) { if t < best_t { best_t = t; best_i = Some(i); } }
        }
        best_i
    }

    /// Ray-sphere pick against builder atoms. Returns dense index (0..n_live_atoms).
    fn pick_builder_atom(&self, mouse: Vec2) -> Option<usize> {
        let (ro, rd) = self.cam.screen_ray(mouse, self.window_size.0, self.window_size.1);
        let mut best_t = f32::MAX; let mut best_i = None;
        for (i, (_, ad)) in self.builder.iter_atoms().enumerate() {
            let p = Vec3::new(ad.pos.x as f32, ad.pos.y as f32, ad.pos.z as f32);
            if let Some(t) = ray_sphere(ro, rd, p, 0.25f32) { if t < best_t { best_t = t; best_i = Some(i); } }
        }
        best_i
    }

    fn do_relax_step(&mut self) {
        if !self.run_relax { return; }
        for _ in 0..self.per_frame {
            let (eb, ea, ed, ei, enb, es) = self.world.eval_forces();
            self.eb = eb; self.ea = ea; self.ed = ed; self.ei = ei; self.enb = enb; self.es = es;
            self.etot = eb + ea + ed + ei + enb + es;
            // Atom dragging via spring force: only in sim mode (not edit mode)
            if !self.show_kekule_editor {
                if let Some(idx) = self.selected {
                    let atom_pos = Vec3::new(self.world.dyn_atoms.atoms.apos.as_slice()[idx].x as f32, self.world.dyn_atoms.atoms.apos.as_slice()[idx].y as f32, self.world.dyn_atoms.atoms.apos.as_slice()[idx].z as f32);
                    let (ray0, hray) = self.cam.screen_ray(self.prev_mouse, self.window_size.0, self.window_size.1);
                    let f_spring = get_force_spring_ray(atom_pos, hray, ray0, self.pick_k as f32);
                    let fapos = self.world.dyn_atoms.fapos.as_mut_slice();
                    fapos[idx].x += f_spring.x as f64; fapos[idx].y += f_spring.y as f64; fapos[idx].z += f_spring.z as f64;
                }
            }
            if self.zero_v_on_opposition {
                let f = self.world.dyn_atoms.fapos.as_slice();
                let v = self.world.dyn_atoms.vapos.as_slice();
                let mut fv = 0.0;
                for i in 0..self.world.natoms() { fv += f[i].dot(v[i]); }
                if fv < 0.0 { self.world.dyn_atoms.clean_velocity(); }
            }
            let cdamp = 1.0 - self.damping;
            for ia in 0..self.world.natoms() { if self.pinned[ia] { continue; } self.world.move_atom_md(ia, self.dt, self.flim, cdamp); }
        }
        self.sync_pos_from_engine();
    }

    fn collect_lines(&self) -> Vec<LineVertex> {
        use molgui::gui::gizmos::{make_bond_segments, make_ring, make_axes, make_crosshair};
        let mut lines = Vec::new();

        if !self.show_kekule_editor {
            // ===== SIMULATION MODE =====
            let ps = self.world.dyn_atoms.atoms.apos.as_slice();
            let natoms = self.world.natoms();

            // --- Bonds ---
            if self.show_bonds {
                const BOND_SEG: i32 = 10;
                let col: [f32; 4] = [0.7, 0.7, 0.7, 1.0];
                for ib in 0..self.world.uff.nbonds as usize {
                    let b = self.world.uff.bon_atoms.as_slice()[ib];
                    let i0 = b[0] as usize; let i1 = b[1] as usize;
                    let p0 = Vec3::new(ps[i0].x as f32, ps[i0].y as f32, ps[i0].z as f32);
                    let p1 = Vec3::new(ps[i1].x as f32, ps[i1].y as f32, ps[i1].z as f32);
                    lines.extend(make_bond_segments(p0, p1, BOND_SEG, col));
                }
            }

            // --- Ports ---
            if self.show_ports {
                let rr = &self.world.rigid_sp3;
                let neigh_bs = self.world.dyn_atoms.atoms.neigh_bs.as_slice();
                let port_col = [1.0f32, 0.5, 0.0, 1.0];
                for i in 0..natoms {
                    let pi = Vec3::new(ps[i].x as f32, ps[i].y as f32, ps[i].z as f32);
                    let bs = neigh_bs[i].as_array();
                    let np = rr.nport[i] as usize;
                    for s in 0..np {
                        let ib = bs[s];
                        if ib < 0 { continue; }
                        let l0 = self.world.uff.bon_params.as_slice()[ib as usize][1];
                        let tip = rr.get_port_tip(ps, i, s, l0);
                        let pt = Vec3::new(tip.x as f32, tip.y as f32, tip.z as f32);
                        lines.push(LineVertex { pos: [pi.x, pi.y, pi.z], col: port_col });
                        lines.push(LineVertex { pos: [pt.x, pt.y, pt.z], col: port_col });
                    }
                }
            }

            // --- Picking highlight ring ---
            if let Some(idx) = self.selected {
                let pos = Vec3::new(ps[idx].x as f32, ps[idx].y as f32, ps[idx].z as f32);
                let r = if self.show_ports { 0.03 } else { self.params.element_radius_vdw(&self.elems[idx]) * ATOM_SCALE };
                let ring_col: [f32; 4] = if self.pinned[idx] { [1.0, 1.0, 0.0, 1.0] } else { [0.0, 1.0, 0.4, 1.0] };
                lines.extend(make_ring(pos, r * 1.6, 16, ring_col));
            }

            // --- Ray from picked atom to cursor ---
            if let Some(idx) = self.selected {
                let atom_pos = Vec3::new(ps[idx].x as f32, ps[idx].y as f32, ps[idx].z as f32);
                let (ray0, _) = self.cam.screen_ray(self.prev_mouse, self.window_size.0, self.window_size.1);
                let red = [1.0f32, 0.0, 0.0, 1.0];
                lines.push(LineVertex { pos: [atom_pos.x, atom_pos.y, atom_pos.z], col: red });
                lines.push(LineVertex { pos: [ray0.x, ray0.y, ray0.z], col: red });
            }
        } else {
            // ===== EDIT MODE =====
            // --- Hex grid reference points (2D lattice dots) ---
            if self.show_hex_grid {
                let grid_col = [0.4f32, 0.4, 0.4, 0.4];
                for p in collect_hex_grid_points(&self.builder.hex_tiles, self.kekule_editor.a_cc) {
                    let sz = 0.03f32;
                    let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
                    lines.extend(make_crosshair(pos, sz, grid_col));
                }
            }

            // --- Ghost hex reference points ---
            if self.show_ghost_hexes {
                let ghost_col = [0.3f32, 0.3, 0.3, 0.3];
                for p in collect_hex_grid_points(&self.builder.hex_tiles, self.kekule_editor.a_cc) {
                    let sz = 0.02f32;
                    let pos = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
                    lines.extend(make_crosshair(pos, sz, ghost_col));
                }
            }

            // --- Builder bonds ---
            let bbond_col = [0.9f32, 0.7, 0.3, 0.8];
            for (a, b) in collect_builder_bonds(&self.builder) {
                lines.push(LineVertex { pos: [a.x as f32, a.y as f32, a.z as f32], col: bbond_col });
                lines.push(LineVertex { pos: [b.x as f32, b.y as f32, b.z as f32], col: bbond_col });
            }

            // --- Builder atoms as colored crosses (larger, element-colored) ---
            for (pos, el) in collect_builder_atoms(&self.builder) {
                let sz = 0.12f32;
                let col = element_color(&el);
                let p = Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32);
                lines.extend(make_crosshair(p, sz, col));
            }

            // --- Picking highlight ring for builder atoms ---
            if let Some(idx) = self.selected {
                if let Some((ah, _)) = self.builder.iter_atoms().nth(idx) {
                    if self.builder.is_atom_alive(ah) {
                        let ad = self.builder.atom(ah);
                        let pos = Vec3::new(ad.pos.x as f32, ad.pos.y as f32, ad.pos.z as f32);
                        let ring_col: [f32; 4] = [0.0, 1.0, 0.4, 1.0];
                        lines.extend(make_ring(pos, 0.25, 16, ring_col));
                    }
                }
            }
        }

        // --- Axes (always visible) ---
        lines.extend(make_axes([0.0, 0.0, 0.0], 1.0));

        // --- Debug cursor: mouse ray origin + direction ---
        if self.show_debug_cursor {
            let (ro, rd) = self.cam.screen_ray(self.mouse_now, self.window_size.0, self.window_size.1);
            let green = [0.0f32, 1.0, 0.0, 1.0];
            let yellow = [1.0f32, 1.0, 0.0, 1.0];
            lines.extend(make_crosshair(ro, 0.5, green));
            let r_end = ro + rd * 20.0;
            lines.push(LineVertex { pos: [ro.x, ro.y, ro.z], col: yellow });
            lines.push(LineVertex { pos: [r_end.x, r_end.y, r_end.z], col: yellow });
        }

        lines
    }

    /// Process all input and physics for one frame; sets dirty flags.
    /// `egui_consumed` is true if egui consumed the event (e.g. click on widget).
    fn update(&mut self, event: &WindowEvent, egui_consumed: bool) -> bool {
        let mut needs_redraw = false;
        match event {
            WindowEvent::Resized(sz) => { self.window_size = (sz.width as f32, sz.height as f32); self.dirty.camera = true; needs_redraw = true; }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32); self.mouse_now = Vec2::new(x, y); self.mouse_delta = self.mouse_now - self.prev_mouse; self.prev_mouse = self.mouse_now;
                // Always allow camera rotation/pan regardless of egui hover state
                if self.trackballing {
                    self.cam.rotate(self.trackball_prev, self.mouse_now, self.window_size.0, self.window_size.1);
                    self.trackball_prev = self.mouse_now; self.dirty.camera = true; needs_redraw = true;
                }
                if self.lmb_down && self.is_shift_down() { self.cam.pan(self.mouse_delta.x, self.mouse_delta.y); self.dirty.camera = true; needs_redraw = true; }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                if *state == ElementState::Pressed {
                    if !egui_consumed { self.lmb_down = true; self.mouse_down = self.mouse_now; }
                } else {
                    self.lmb_down = false;
                    if !egui_consumed {
                        let dpix = (self.mouse_now - self.mouse_down).length();
                        if dpix < 5.0 {
                            if self.show_kekule_editor {
                                // ===== EDIT MODE =====
                                match self.kekule_editor.edit_mode {
                                    EditMode::Select => {
                                        let picked = self.pick_builder_atom(self.mouse_now);
                                        self.selected = if self.selected == picked { None } else { picked };
                                    }
                                    EditMode::HexPaint | EditMode::HexToggle | EditMode::AtomDraw => {
                                        if let Some(pos_ws) = self.mouse_ray_z0(self.mouse_now) {
                                            if self.kekule_editor.on_click(&mut self.builder, pos_ws) {
                                                self.edit_from_builder = true;
                                                println!("Builder modified: {}", builder_summary(&self.builder));
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            } else {
                                // ===== SIMULATION MODE =====
                                self.selected = if self.selected == self.pick_atom(self.mouse_now) { None } else { self.pick_atom(self.mouse_now) };
                            }
                            needs_redraw = true;
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                if *state == ElementState::Pressed {
                    self.trackballing = true; self.trackball_prev = self.mouse_now;
                } else {
                    self.trackballing = false;
                    if !egui_consumed {
                        if self.show_kekule_editor {
                            match self.kekule_editor.edit_mode {
                                EditMode::HexPaint | EditMode::HexToggle => {
                                    if let Some(pos_ws) = self.mouse_ray_z0(self.mouse_now) {
                                        if self.kekule_editor.on_right_click(&mut self.builder, pos_ws) {
                                            self.edit_from_builder = true;
                                            println!("Builder modified (right-click): {}", builder_summary(&self.builder));
                                        }
                                    }
                                }
                                EditMode::Select | EditMode::AtomDraw => { self.selected = None; }
                                _ => {}
                            }
                        } else {
                            self.selected = None;
                        }
                    }
                    needs_redraw = true;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Always allow zoom, even when hovering over egui widgets
                let dy = match delta { MouseScrollDelta::LineDelta(_, y) => *y, MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01 };
                self.cam.zoom(dy); self.dirty.camera = true; needs_redraw = true;
            }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Named(NamedKey::Space), state: ElementState::Pressed, .. }, .. } => { self.run_relax = !self.run_relax; println!("relax = {}", self.run_relax); needs_redraw = true; }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Character(c), state: ElementState::Pressed, .. }, .. } => {
                if !egui_consumed {
                    match c.as_str() {
                        "h" | "H" => { self.show_help = !self.show_help; needs_redraw = true; }
                        "b" | "B" => { self.show_bonds = !self.show_bonds; needs_redraw = true; }
                        "s" | "S" => { self.show_surface = !self.show_surface; needs_redraw = true; }
                        "g" | "G" => { self.show_groups = !self.show_groups; needs_redraw = true; }
                        "t" | "T" => { self.show_ports = !self.show_ports; needs_redraw = true; }
                        "k" | "K" => { self.show_labels = !self.show_labels; needs_redraw = true; }
                        "d" | "D" => { self.show_debug_cursor = !self.show_debug_cursor; needs_redraw = true; }
                        "p" | "P" => { if let Some(idx) = self.selected { self.pinned[idx] = !self.pinned[idx]; } needs_redraw = true; }
                        "c" | "C" => { self.cam = TrackballCam::new(Vec3::new(0.0, 1.0, 0.0), 6.0); self.dirty.camera = true; needs_redraw = true; }
                        "l" | "L" => { self.label_mode = match self.label_mode { LabelMode::None => LabelMode::AtomNumber, LabelMode::AtomNumber => LabelMode::AtomType, LabelMode::AtomType => LabelMode::Charge, LabelMode::Charge => LabelMode::ElementName, LabelMode::ElementName => LabelMode::None }; println!("label_mode = {:?}", self.label_mode); needs_redraw = true; }
                        "e" | "E" => { self.show_kekule_editor = !self.show_kekule_editor; println!("show_kekule_editor = {}", self.show_kekule_editor); needs_redraw = true; }
                        "f" | "F" => { self.world.bonded_mode = match self.world.bonded_mode { BondedFFMode::Uff => BondedFFMode::RigidSp3, BondedFFMode::RigidSp3 => BondedFFMode::Uff }; println!("bonded_mode = {:?}", self.world.bonded_mode); needs_redraw = true; }
                        "1" => { self.kekule_editor.set_edit_mode(EditMode::Select); println!("edit_mode = Select"); needs_redraw = true; }
                        "2" => { self.kekule_editor.set_edit_mode(EditMode::HexPaint); println!("edit_mode = HexPaint"); needs_redraw = true; }
                        "3" => { self.kekule_editor.set_edit_mode(EditMode::HexToggle); println!("edit_mode = HexToggle"); needs_redraw = true; }
                        "4" => { self.kekule_editor.set_edit_mode(EditMode::AtomDraw); println!("edit_mode = AtomDraw"); needs_redraw = true; }
                        "n" | "N" => { if self.world.nonbonded.is_some() { self.world.nonbonded = None; println!("nonbonded = None"); } else { let mut nb = NonBondedFF::new(self.world.natoms()); nb.set_cutoff(8.0); self.world.nonbonded = Some(nb); println!("nonbonded = LJ+Coulomb"); } needs_redraw = true; }
                        "m" | "M" => { if self.world.surface.is_some() { self.world.surface = None; println!("surface = None"); } else { self.world.setup_nacl_surface(LATTICE_A, SURFACE_Z0 as f64, BETA_CHARGE, BETA_MORSE_RATIO, Q_AMP, PLQ_AMP); println!("surface = NaCl"); } self.rebuild_surface_cache(); needs_redraw = true; }
                        "[" => { self.pick_k *= 0.8; println!("pick_k = {}", self.pick_k); }
                        "]" => { self.pick_k *= 1.25; println!("pick_k = {}", self.pick_k); }
                        "-" => { self.per_frame = (self.per_frame / 2).max(1); println!("per_frame = {}", self.per_frame); }
                        "=" | "+" => { self.per_frame = (self.per_frame * 2).min(2000); println!("per_frame = {}", self.per_frame); }
                        _ => {}
                    }
                }
            }
            WindowEvent::KeyboardInput { event: KeyEvent { logical_key: Key::Named(NamedKey::Escape), state: ElementState::Pressed, .. }, .. } => { self.selected = None; needs_redraw = true; }
            _ => {}
        }
        needs_redraw
    }

    fn is_shift_down(&self) -> bool { false } // TODO: track modifier state if needed

    /// Compute world-space intersection of mouse ray with Z=0 plane.
    fn mouse_ray_z0(&self, mouse: Vec2) -> Option<Vec3d> {
        let (ro, rd) = self.cam.screen_ray(mouse, self.window_size.0, self.window_size.1);
        let ro = Vec3d::new(ro.x as f64, ro.y as f64, ro.z as f64);
        let rd = Vec3d::new(rd.x as f64, rd.y as f64, rd.z as f64);
        if rd.z.abs() < 1e-6 { return None; }
        let t = -ro.z / rd.z;
        if t < 0.0 { return None; }
        Some(Vec3d::new(ro.x + rd.x * t, ro.y + rd.y * t, 0.0))
    }

    /// Rebuild full instance array (pos, radius, color) from current world + elems.
    /// Needed after Bake to Sim when natoms or element types change.
    fn rebuild_instances(&mut self) {
        let natoms = self.world.natoms();
        if natoms > self.renderer.max_atoms() {
            let w = self.config.width;
            let h = self.config.height;
            let mut renderer = ImpostorRenderer::new(self.device.clone(), self.queue.clone(), natoms.max(1), self.config.format);
            renderer.set_target_size(w, h);
            self.renderer = renderer;
        }
        self.instances.clear();
        self.instances.reserve(natoms);
        let ps = self.world.dyn_atoms.atoms.apos.as_slice();
        for i in 0..natoms {
            let el = self.elems.get(i).map(|s| s.as_str()).unwrap_or("C");
            let col = self.params.element_color_f32(el);
            let r = self.params.element_radius_vdw(el) * ATOM_SCALE;
            let p = &ps[i];
            self.instances.push(AtomInstance { pos: [p.x as f32, p.y as f32, p.z as f32], radius: r, color: col, _pad: 0.0 });
        }
        self.renderer.set_atoms(&self.instances);
    }

    /// Upload GPU data only when dirty. Called before render.
    fn prepare(&mut self) {
        if self.dirty.atoms {
            let natoms = self.world.natoms();
            if self.instances.len() != natoms {
                self.rebuild_instances();
            } else {
                let ps = self.world.dyn_atoms.atoms.apos.as_slice();
                for i in 0..natoms { let p = &ps[i]; self.instances[i].pos = [p.x as f32, p.y as f32, p.z as f32]; }
                self.renderer.set_atoms(&self.instances);
            }
            self.dirty.atoms = false;
        }
        if self.dirty.camera {
            // camera_data is computed fresh each render; nothing to persist here
            self.dirty.camera = false;
        }
    }
}

/// Map surface potential to RGBA (blue-white-red diverging).
fn potential_color(pot: f32) -> [u8; 4] {
    let vmax = 1.0;
    let t = (pot / vmax).clamp(-1.0, 1.0);
    if t < 0.0 {
        let s = (1.0 + t) as f32; // s in [0,1]
        [(255.0 * s) as u8, (255.0 * s) as u8, 255, 255]
    } else {
        let s = (1.0 - t) as f32; // s in [0,1]
        [255, (255.0 * s) as u8, (255.0 * s) as u8, 255]
    }
}

impl App {
    /// Sample the surface potential on a grid aligned with the lattice vectors and bake into a wgpu texture.
    fn rebuild_surface_cache(&mut self) {
        self.surface_texture = None;
        self.surface_origin = [0.0; 3];
        self.surface_u = [0.0; 3];
        self.surface_v = [0.0; 3];
        let Some(ref surf) = self.world.surface else { return };
        let z0 = SURFACE_Z0;
        let n = SURFACE_GRID_N;
        let dummy_req = [0.0, 0.0, 1.0, 0.0]; // unit charge, no Pauli/London
        let dummy_plq = surfff::SurfaceFolded::req2plq(dummy_req, 2.0);

        // Build parallelogram from lattice vectors × replicas
        let ax = surf.ax as f32; let ay = surf.ay as f32;
        let bx = surf.bx as f32; let by = surf.by as f32;
        let la = (ax * ax + ay * ay).sqrt();
        let lb = (bx * bx + by * by).sqrt();
        let n_rep = (16.0 / la.max(lb)).ceil().max(2.0).min(6.0) as f32;
        let u_edge = [ax * n_rep, ay * n_rep, 0.0];
        let v_edge = [bx * n_rep, by * n_rep, 0.0];
        let origin = [-0.5 * (u_edge[0] + v_edge[0]), -0.5 * (u_edge[1] + v_edge[1]), z0];
        self.surface_origin = origin;
        self.surface_u = u_edge;
        self.surface_v = v_edge;

        let w = (n + 1) as usize;
        let h = (n + 1) as usize;
        let mut pixels = vec![0u8; w * h * 4];

        for iy in 0..=n {
            let t = iy as f32 / n as f32;
            for ix in 0..=n {
                let s = ix as f32 / n as f32;
                let x = origin[0] + u_edge[0] * s + v_edge[0] * t;
                let y = origin[1] + u_edge[1] * s + v_edge[1] * t;
                let pos = Vec3d::new(x as f64, y as f64, z0 as f64);
                let (e, _) = surf.eval_atom(pos, dummy_plq);
                let pot = e as f32;
                let color = potential_color(pot);
                let row = (n - iy) as usize; // flip Y for texture orientation
                let pix = (row * w + ix as usize) * 4;
                pixels[pix..pix + 4].copy_from_slice(&color);
            }
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("surface_potential"),
            size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w as u32 * 4), rows_per_image: Some(h as u32) },
            wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
        );
        self.surface_texture = Some(texture);
    }

    fn render(&mut self) {
        let frame = self.surface.get_current_texture();
        let surface_texture = match frame {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout => return,
            wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Outdated => { self.surface.configure(&self.device, &self.config); return; }
            wgpu::CurrentSurfaceTexture::Lost => { self.surface.configure(&self.device, &self.config); return; }
            wgpu::CurrentSurfaceTexture::Validation => return,
        };
        let view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let cam = self.cam.camera_data(self.config.width, self.config.height);
        if self.show_kekule_editor {
            self.renderer.set_atoms(&[]);
        }
        self.renderer.render(&view, wgpu::Color { r: 0.08, g: 0.08, b: 0.08, a: 1.0 }, &cam);
        if self.show_kekule_editor {
            self.renderer.set_atoms(&self.instances);
        }

        // --- Surface potential textured quad ---
        if self.show_surface {
            if let Some(ref tex) = self.surface_texture {
                let tex_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("surface_encoder") });
                let depth_view = self.renderer.depth_view();
                self.surface_renderer.render(&mut encoder, &view, depth_view, &cam, &tex_view, self.surface_origin, self.surface_u, self.surface_v);
                self.queue.submit(std::iter::once(encoder.finish()));
            }
        }

        let lines = self.collect_lines();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("line_egui_encoder") });
        let depth_view = self.renderer.depth_view();
        self.line_renderer.render(&mut encoder, &view, depth_view, &cam, &lines);

        // --- egui overlay ---
        let mut raw_input = self.egui_state.take_egui_input(&self.window);
        // Clipboard bridge: inject Cut/Copy/Paste events (replaces egui-winit's clipboard feature)
        let mods = raw_input.modifiers;
        inject_cut_copy_if_needed(&mut raw_input.events, mods);
        inject_paste_if_needed(&mut raw_input.events, mods, &mut self.clipboard);
        let egui_ctx = self.egui_ctx.clone();
        let full_output = egui_ctx.run(raw_input, |ctx| { self.draw_egui(ctx); });
        // Clipboard bridge: write CopyText commands to OS clipboard
        handle_output_commands(&full_output.platform_output, &mut self.clipboard);
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set { self.egui_renderer.update_texture(&*self.device, &self.queue, *id, image_delta); }
        for id in &full_output.textures_delta.free { self.egui_renderer.free_texture(id); }
        let screen_descriptor = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.config.width, self.config.height], pixels_per_point: full_output.pixels_per_point };
        self.egui_renderer.update_buffers(&*self.device, &self.queue, &mut encoder, &tris, &screen_descriptor);
        {
            let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            self.egui_renderer.render(&mut egui_pass.forget_lifetime(), &tris, &screen_descriptor);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();
    }

    fn draw_egui(&mut self, ctx: &egui::Context) {
        let w = ctx.screen_rect().width();
        let h = ctx.screen_rect().height();

        // Title + Energy (top-left)
        egui::Window::new("SurfMol")
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
            .resizable(false)
            .title_bar(false)
            .frame(egui::Frame::none().fill(egui::Color32::from_rgba_premultiplied(0, 0, 0, 180)))
            .show(ctx, |ui| {
                ui.label(egui::RichText::new("Molecule-on-Surface Viewer").size(20.0).color(egui::Color32::WHITE));
                ui.separator();
                ui.label(egui::RichText::new(format!("Etotal = {:10.4} eV", self.etot)).size(16.0));
                ui.label(egui::RichText::new(format!("  bond={:8.3} angle={:8.3} dihed={:8.3}", self.eb, self.ea, self.ed)).size(14.0).color(egui::Color32::GRAY));
                ui.label(egui::RichText::new(format!("  inv ={:8.3} nb  ={:8.3} surf={:8.3}", self.ei, self.enb, self.es)).size(14.0).color(egui::Color32::GRAY));
            });

        // Selected atom info (top-right)
        if let Some(idx) = self.selected {
            egui::Window::new("Atom Info")
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
                .resizable(false)
                .title_bar(false)
                .frame(egui::Frame::none().fill(egui::Color32::from_rgba_premultiplied(0, 0, 0, 180)))
                .show(ctx, |ui| {
                    if self.show_kekule_editor {
                        // Builder atom info
                        if let Some((_, ad)) = self.builder.iter_atoms().nth(idx) {
                            ui.label(egui::RichText::new(format!("Atom {}: {}", idx, ad.element)).size(18.0).color(egui::Color32::YELLOW));
                            ui.label(egui::RichText::new(format!("pos: {:.3} {:.3} {:.3}", ad.pos.x, ad.pos.y, ad.pos.z)).size(14.0).color(egui::Color32::GRAY));
                            let r = self.params.element_radius_vdw(&ad.element);
                            ui.label(egui::RichText::new(format!("RvdW = {:.3} Å", r)).size(14.0).color(egui::Color32::GRAY));
                            if ad.is_h_cap { ui.label(egui::RichText::new("[H-cap]").size(14.0).color(egui::Color32::from_rgb(255, 160, 0))); }
                        }
                    } else {
                        // Sim atom info
                        ui.label(egui::RichText::new(format!("Atom {}: {}", idx, self.elems[idx])).size(18.0).color(egui::Color32::YELLOW));
                        let pin_text = if self.pinned[idx] { "[PINNED]  Press P to unpin" } else { "Press P to pin" };
                        ui.label(egui::RichText::new(pin_text).size(14.0).color(if self.pinned[idx] { egui::Color32::from_rgb(255, 160, 0) } else { egui::Color32::GRAY }));
                        let pos = self.world.dyn_atoms.atoms.apos.as_slice()[idx];
                        ui.label(egui::RichText::new(format!("pos: {:.3} {:.3} {:.3}", pos.x, pos.y, pos.z)).size(14.0).color(egui::Color32::GRAY));
                        let r = self.params.element_radius_vdw(&self.elems[idx]);
                        ui.label(egui::RichText::new(format!("RvdW = {:.3} Å", r)).size(14.0).color(egui::Color32::GRAY));
                    }
                });
        }

        // Settings panel (right side)
        egui::Window::new("Settings")
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 140.0))
            .resizable(false)
            .title_bar(true)
            .frame(egui::Frame::window(&ctx.style()))
            .show(ctx, |ui| {
                ui.heading("Physics");
                ui.horizontal(|ui| {
                    ui.label("Iters/frame:");
                    ui.add(egui::DragValue::new(&mut self.per_frame).speed(1).clamp_range(1..=2000));
                });
                ui.horizontal(|ui| {
                    ui.label("Time step dt:");
                    ui.add(egui::DragValue::new(&mut self.dt).speed(0.001).clamp_range(0.0001..=0.5));
                });
                ui.horizontal(|ui| {
                    ui.label("Damping:");
                    ui.add(egui::DragValue::new(&mut self.damping).speed(0.01).clamp_range(0.0..=1.0));
                });
                ui.checkbox(&mut self.zero_v_on_opposition, "Zero V when F·V < 0 (3N)");
                ui.separator();
                ui.heading("Display");
                ui.horizontal(|ui| {
                    ui.label("Labels:");
                    egui::ComboBox::from_id_source("label_mode_combo")
                        .width(120.0)
                        .selected_text(match self.label_mode {
                            LabelMode::None => "None",
                            LabelMode::AtomNumber => "Atom Number",
                            LabelMode::AtomType => "Atom Type",
                            LabelMode::Charge => "Charge",
                            LabelMode::ElementName => "Element Name",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.label_mode, LabelMode::None, "None");
                            ui.selectable_value(&mut self.label_mode, LabelMode::AtomNumber, "Atom Number");
                            ui.selectable_value(&mut self.label_mode, LabelMode::AtomType, "Atom Type");
                            ui.selectable_value(&mut self.label_mode, LabelMode::Charge, "Charge");
                            ui.selectable_value(&mut self.label_mode, LabelMode::ElementName, "Element Name");
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Bonded FF:");
                    egui::ComboBox::from_id_source("bonded_mode_combo")
                        .width(120.0)
                        .selected_text(match self.world.bonded_mode {
                            BondedFFMode::Uff => "UFF",
                            BondedFFMode::RigidSp3 => "Rigid sp3",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.world.bonded_mode, BondedFFMode::Uff, "UFF");
                            ui.selectable_value(&mut self.world.bonded_mode, BondedFFMode::RigidSp3, "Rigid sp3");
                        });
                });
                ui.separator();
                let nb_str = if self.world.nonbonded.is_some() { "LJ+Coulomb" } else { "None" };
                ui.label(egui::RichText::new(format!("Non-bonded (N): {}", nb_str)).size(14.0).color(egui::Color32::YELLOW));
                let sf_str = if self.world.surface.is_some() { "NaCl" } else { "None" };
                ui.label(egui::RichText::new(format!("Surface (M): {}", sf_str)).size(14.0).color(egui::Color32::YELLOW));
            });

        // Kekule Editor panel (left side, below title)
        if self.show_kekule_editor {
            egui::Window::new("Kekule Editor")
                .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 120.0))
                .resizable(false)
                .title_bar(true)
                .frame(egui::Frame::window(&ctx.style()))
                .show(ctx, |ui| {
                    ui.heading("Edit Mode");
                    ui.horizontal(|ui| {
                        if ui.selectable_label(self.kekule_editor.edit_mode == EditMode::Select, "Select (1)").clicked() { self.kekule_editor.set_edit_mode(EditMode::Select); }
                        if ui.selectable_label(self.kekule_editor.edit_mode == EditMode::HexPaint, "HexPaint (2)").clicked() { self.kekule_editor.set_edit_mode(EditMode::HexPaint); }
                        if ui.selectable_label(self.kekule_editor.edit_mode == EditMode::HexToggle, "HexToggle (3)").clicked() { self.kekule_editor.set_edit_mode(EditMode::HexToggle); }
                        if ui.selectable_label(self.kekule_editor.edit_mode == EditMode::AtomDraw, "AtomDraw (4)").clicked() { self.kekule_editor.set_edit_mode(EditMode::AtomDraw); }
                    });
                    ui.separator();
                    ui.heading("Atom Type");
                    ui.horizontal(|ui| {
                        if ui.selectable_label(self.kekule_editor.atom_type == AtomType::C, "C").clicked() { self.kekule_editor.set_atom_type(AtomType::C); }
                        if ui.selectable_label(self.kekule_editor.atom_type == AtomType::N, "N").clicked() { self.kekule_editor.set_atom_type(AtomType::N); }
                        if ui.selectable_label(self.kekule_editor.atom_type == AtomType::O, "O").clicked() { self.kekule_editor.set_atom_type(AtomType::O); }
                        if ui.selectable_label(self.kekule_editor.atom_type == AtomType::H, "H").clicked() { self.kekule_editor.set_atom_type(AtomType::H); }
                    });
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.kekule_editor.auto_h_cap, "Auto H");
                        ui.checkbox(&mut self.kekule_editor.auto_bonds, "Auto Bonds");
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.kekule_editor.grid_mode, "Grid Snap");
                        ui.checkbox(&mut self.show_hex_grid, "Hex Grid");
                        ui.checkbox(&mut self.show_ghost_hexes, "Ghost Hexes");
                    });
                    ui.separator();
                    ui.label(format!("Builder: {}", builder_summary(&self.builder)));
                    ui.horizontal(|ui| {
                        if ui.button("Cleanup Dead").clicked() { self.builder.cleanup_dead(); self.edit_from_builder = true; }
                        if ui.button("Bake to Sim").clicked() {
                            let had_nonbonded = self.world.nonbonded.is_some();
                            let had_surface = self.world.surface.is_some();
                            self.builder.cleanup_dead();
                            let top = self.builder.bake();
                            let elems = self.builder.bake_elements();
                            self.world = MolWorld::from_topology(&top);
                            self.world.make_neigh_bs();
                            self.world.bake_angle_neighs();
                            self.world.bake_dihedral_neighs();
                            self.world.bake_inversion_neighs();
                            self.world.map_atom_interactions();
                            self.elems = elems;
                            let natoms = self.world.natoms();
                            self.pinned = vec![false; natoms];
                            let neighs_q4 = self.world.dyn_atoms.neighs();
                            let neighs: Vec<[i32; 4]> = neighs_q4.iter().map(|n| [n.x, n.y, n.z, n.w]).collect();
                            self.uff_types = assign_uff::assign_uff_types(&self.elems, &neighs);
                            self.charges = vec![0.0; natoms];
                            self.selected = None;
                            // Forcefield rebuild (MUST be done after every bake)
                            if had_nonbonded {
                                self.world.nonbonded = Some(NonBondedFF::new(natoms));
                                self.world.nonbonded.as_mut().unwrap().make_second_neighs(neighs_q4, natoms);
                                self.world.nonbonded.as_mut().unwrap().set_cutoff(8.0);
                            }
                            let has_sp2 = self.uff_types.iter().any(|t| matches!(t.as_str(), "C_R"|"C_2"|"N_R"|"O_2"|"O_R"));
                            if has_sp2 { self.world.bonded_mode = BondedFFMode::Uff; }
                            self.world.rigid_sp3.set_port_geometry_from_types(&self.uff_types);
                            let have_params = !self.params.elements.is_empty() && !self.params.atom_types.is_empty() && !self.params.bonds.is_empty() && !self.params.angles.is_empty();
                            if have_params {
                                if let Some(ref mut nb) = self.world.nonbonded {
                                    for i in 0..natoms {
                                        let t = self.uff_types[i].as_str();
                                        let mut req = get_reqh(&self.params, t);
                                        if self.charges[i] != 0.0 { req[2] = self.charges[i]; }
                                        nb.reqs.as_mut_slice()[i] = req;
                                    }
                                    nb.make_plqs(2.0);
                                }
                                for ib in 0..self.world.uff.nbonds as usize {
                                    let b = self.world.uff.bon_atoms.as_slice()[ib];
                                    let ia = b[0] as usize; let ja = b[1] as usize;
                                    let a = self.elems[ia].as_str();
                                    let b = self.elems[ja].as_str();
                                    if let Some(bp) = self.params.get_bond_param(a, b, 1) { self.world.uff.bon_params.as_mut_slice()[ib] = [bp.k, bp.l0]; } else { panic!("missing bond param for {}-{} order=1", a, b); }
                                }
                                for ia in 0..self.world.uff.nangles as usize {
                                    let ang = self.world.uff.ang_atoms.as_slice()[ia];
                                    let i0 = ang[0] as usize; let i1 = ang[1] as usize; let i2 = ang[2] as usize;
                                    let a = self.elems[i0].as_str();
                                    let b = self.elems[i1].as_str();
                                    let c = self.elems[i2].as_str();
                                    let ap = self.params.get_angle_param(a, b, c).unwrap_or_else(|| panic!("missing angle param for {}-{}-{}", a, b, c));
                                    let th0 = ap.a0.to_radians();
                                    let ct = th0.cos();
                                    let st2 = 1.0 - ct * ct;
                                    assert!(st2 > 1e-12, "invalid angle theta0={} deg leads to sin^2(theta0)~0", ap.a0);
                                    let c2 = 1.0 / (4.0 * st2);
                                    let c1 = -4.0 * c2 * ct;
                                    let c0 = c2 * (2.0 * ct * ct + 1.0);
                                    self.world.uff.ang_params.as_mut_slice()[ia] = [ap.k, c0, c1, c2, 0.0];
                                }
                                for id in 0..self.world.uff.ndihedrals as usize {
                                    let d = self.world.uff.dih_atoms.as_slice()[id];
                                    let a = self.uff_types[d.x as usize].as_str();
                                    let b = self.uff_types[d.y as usize].as_str();
                                    let c = self.uff_types[d.z as usize].as_str();
                                    let e = self.uff_types[d.w as usize].as_str();
                                    if let Some(dp) = self.params.get_dihedral_param(a, b, c, e, 1) {
                                        let a0 = dp.a0.to_radians();
                                        let n = dp.n as f64;
                                        let phase = n * a0;
                                        let s = phase.sin().abs();
                                        if s > 1e-3 { panic!("dihedral phase not supported by current Uff dihedral form: {}-{}-{}-{} a0={}deg n={} => n*a0={}deg", a, b, c, e, dp.a0, dp.n, phase.to_degrees()); }
                                        let dsign = if phase.cos() >= 0.0 { 1.0 } else { -1.0 };
                                        self.world.uff.dih_params.as_mut_slice()[id] = [dp.k, dsign, dp.n as f64];
                                    } else {
                                        self.world.uff.dih_params.as_mut_slice()[id] = [0.0, 1.0, 3.0];
                                    }
                                }
                                for ii in 0..self.world.uff.ninversions as usize {
                                    let inv = self.world.uff.inv_atoms.as_slice()[ii];
                                    let ic = inv.x as usize;
                                    let t = self.uff_types[ic].as_str();
                                    if matches!(t, "C_R"|"C_2"|"N_R"|"O_2"|"O_R") { self.world.uff.inv_params.as_mut_slice()[ii] = [50.0, 1.0, -1.0, 0.0]; } else { self.world.uff.inv_params.as_mut_slice()[ii] = [0.0, 1.0, -1.0, 0.0]; }
                                }
                            }
                            if had_surface {
                                self.world.setup_nacl_surface(LATTICE_A, SURFACE_Z0 as f64, BETA_CHARGE, BETA_MORSE_RATIO, Q_AMP, PLQ_AMP);
                            }
                            self.world.update_hneigh();
                            self.rebuild_instances();
                            self.dirty.atoms = true;
                            println!("Baked to sim: {} atoms, {} bonds", self.world.natoms(), top.bonds.len());
                            self.show_kekule_editor = false; // switch to sim mode after bake
                        }
                        if ui.button("Export XYZ").clicked() {
                            let xyz = export_xyz(&self.builder);
                            println!("=== XYZ Export ===\n{}", xyz);
                        }
                    });
                    ui.separator();
                    ui.heading("Ribbon");
                    ui.horizontal(|ui| {
                        ui.label("Rows:");
                        ui.add(egui::DragValue::new(&mut self.kekule_editor.ribbon_rows).speed(1).clamp_range(1..=20));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Bottom:");
                        ui.text_edit_singleline(&mut self.kekule_editor.ribbon_bottom);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Top:");
                        ui.text_edit_singleline(&mut self.kekule_editor.ribbon_top);
                    });
                    if ui.button("Generate Ribbon").clicked() {
                        let pass_bottom = molgui::gui::kekule_editor::parse_passivation_string(&self.kekule_editor.ribbon_bottom);
                        let pass_top = molgui::gui::kekule_editor::parse_passivation_string(&self.kekule_editor.ribbon_top);
                        let a_cc = self.kekule_editor.a_cc;
                        let scale_x = 1.0;
                        molgui::gui::kekule_editor::build_zigzag_ribbon(&mut self.builder, self.kekule_editor.ribbon_rows, pass_bottom.len() as i32, &pass_bottom, &pass_top, a_cc, scale_x);
                        self.edit_from_builder = true;
                        println!("Generated ribbon: {}", builder_summary(&self.builder));
                    }
                });
        }

        // Help (bottom-left)
        if self.show_help {
            egui::Window::new("Help")
                .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(10.0, -10.0))
                .resizable(false)
                .title_bar(false)
                .frame(egui::Frame::none().fill(egui::Color32::from_rgba_premultiplied(0, 0, 0, 180)))
                .show(ctx, |ui| {
                    ui.label(egui::RichText::new("Controls:").size(16.0).color(egui::Color32::WHITE));
                    let help = [
                        "LMB click atom     -> pick/unpick (Select mode)",
                        "LMB click          -> hex paint/toggle/atom (Edit mode)",
                        "RMB click          -> unpick (Select) / remove hex (Edit)",
                        "Shift+LMB drag     -> pan camera",
                        "RMB drag           -> rotate camera",
                        "Scroll             -> zoom",
                        "SPACE              -> start/stop relaxation",
                        "1/2/3/4            -> Select/HexPaint/HexToggle/AtomDraw",
                        "E                  -> toggle Kekule editor panel",
                        "P                  -> pin/unpin picked atom",
                        "S                  -> toggle surface",
                        "B                  -> toggle bonds",
                        "H                  -> toggle help",
                        "ESC                -> unpick",
                        "C                  -> reset camera",
                        "G                  -> toggle group AABBs",
                        "T                  -> toggle ports",
                        "F                  -> toggle bonded FF",
                        "L                  -> cycle label mode",
                        "K                  -> toggle labels",
                        "D                  -> toggle debug cursor",
                        "N                  -> toggle non-bonded",
                        "M                  -> toggle surface FF",
                    ];
                    for line in help {
                        ui.label(egui::RichText::new(line).size(12.0).color(egui::Color32::GRAY));
                    }
                });
        } else {
            egui::Window::new("Hint")
                .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(10.0, -10.0))
                .resizable(false)
                .title_bar(false)
                .frame(egui::Frame::none())
                .show(ctx, |ui| {
                    ui.label(egui::RichText::new("Press H for help").size(14.0).color(egui::Color32::DARK_GRAY));
                });
        }

        // Status (bottom)
        let relax_str = if self.run_relax { "ON  (press SPACE to pause)" } else { "OFF (press SPACE to run)" };
        let relax_col = if self.run_relax { egui::Color32::GREEN } else { egui::Color32::GRAY };
        egui::Window::new("Status")
            .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -10.0))
            .resizable(false)
            .title_bar(false)
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                ui.label(egui::RichText::new(format!("Relaxation: {}", relax_str)).size(16.0).color(relax_col));
            });

        // Screen-space labels next to atoms
        if self.show_labels {
            let cam = self.cam.camera_data(self.config.width, self.config.height);
            let vp = glam::Mat4::from_cols_array_2d(&cam.view_proj);
            let w = self.config.width as f32;
            let h = self.config.height as f32;
            if self.show_kekule_editor {
                // Builder atom labels
                for (i, (_, ad)) in self.builder.iter_atoms().enumerate() {
                    let world_pos = glam::Vec4::new(ad.pos.x as f32, ad.pos.y as f32, ad.pos.z as f32, 1.0);
                    let clip = vp * world_pos;
                    if clip.w <= 0.0 { continue; }
                    let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
                    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 { continue; }
                    let sp = glam::Vec2::new((ndc.x + 1.0) * 0.5 * w, (1.0 - (ndc.y + 1.0) * 0.5) * h);
                    let txt = match self.label_mode {
                        LabelMode::None => continue,
                        LabelMode::AtomNumber => format!("{}", i),
                        LabelMode::AtomType => ad.element.clone(),
                        LabelMode::Charge => format!("{:.2}", 0.0),
                        LabelMode::ElementName => ad.element.clone(),
                    };
                    egui::Area::new(egui::Id::new(("builder_label", i)))
                        .fixed_pos(egui::pos2(sp.x, sp.y + 5.0))
                        .show(ctx, |ui| {
                            ui.add(egui::Label::new(egui::RichText::new(&txt).size(12.0).color(egui::Color32::WHITE)).extend());
                        });
                }
            } else {
                // Sim atom labels
                for i in 0..self.world.natoms() {
                    let p = self.world.dyn_atoms.atoms.apos.as_slice()[i];
                    let world_pos = glam::Vec4::new(p.x as f32, p.y as f32, p.z as f32, 1.0);
                    let clip = vp * world_pos;
                    if clip.w <= 0.0 { continue; }
                    let ndc = glam::Vec2::new(clip.x / clip.w, clip.y / clip.w);
                    if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 { continue; }
                    let sp = glam::Vec2::new((ndc.x + 1.0) * 0.5 * w, (1.0 - (ndc.y + 1.0) * 0.5) * h);
                    let txt = match self.label_mode {
                        LabelMode::None => continue,
                        LabelMode::AtomNumber => format!("{}", i),
                        LabelMode::AtomType => self.uff_types.get(i).cloned().unwrap_or_else(|| self.elems[i].clone()),
                        LabelMode::Charge => format!("{:.2}", self.charges.get(i).unwrap_or(&0.0)),
                        LabelMode::ElementName => self.elems[i].clone(),
                    };
                    egui::Area::new(egui::Id::new(("atom_label", i)))
                        .fixed_pos(egui::pos2(sp.x, sp.y + 5.0))
                        .show(ctx, |ui| {
                            ui.add(egui::Label::new(egui::RichText::new(&txt).size(12.0).color(egui::Color32::WHITE)).extend());
                        });
                }
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window_attrs = Window::default_attributes()
        .with_title("SurfMol (wgpu)")
        .with_inner_size(winit::dpi::PhysicalSize::new(1800, 1000));
    let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
    let mut app = App::new(window.clone());
    let mut pending_redraw = true;
    let mut last_instant = std::time::Instant::now();
    app.window.request_redraw();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => elwt.exit(),
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                pending_redraw = false;
                let now = std::time::Instant::now();
                let dt = (now - last_instant).as_secs_f32();
                last_instant = now;
                app.cam.update(dt);
                if app.run_relax { app.do_relax_step(); }
                app.prepare();
                app.render();
                // If egui animations/widgets still need frames, keep redrawing
                if app.egui_ctx.has_requested_repaint() {
                    pending_redraw = true;
                    app.window.request_redraw();
                }
            }
            Event::WindowEvent { event, .. } => {
                let egui_response = app.egui_state.on_window_event(&app.window, &event);
                if let WindowEvent::Resized(sz) = event { app.resize(sz); }
                let needs_redraw = app.update(&event, egui_response.consumed);
                if needs_redraw || egui_response.repaint {
                    pending_redraw = true;
                    app.window.request_redraw();
                }
            }
            Event::AboutToWait => {
                // Always request redraw when simulating or when egui needs animation frames.
                // This ensures combo boxes, hover effects, and cursor animations stay smooth.
                if app.run_relax || pending_redraw || app.egui_ctx.has_requested_repaint() {
                    app.window.request_redraw();
                }
            }
            _ => {}
        }
    }).unwrap();
}
