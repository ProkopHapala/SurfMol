use std::path::PathBuf;

use eframe::egui;
use surfmol_common::xyz::read_xyz;
use surfmol_topology::params::Params;
use surfmol_apps::thumbnailer::MolThumbnailer;

// ------------------------------------------------------------------
// Molecule entry
// ------------------------------------------------------------------

struct MolEntry {
    path: PathBuf,
    name: String,
    natoms: usize,
    thumbnail: Option<egui::ColorImage>,
    texture: Option<egui::TextureHandle>,
}

// ------------------------------------------------------------------
// App state
// ------------------------------------------------------------------

struct MolBrowserApp {
    entries: Vec<MolEntry>,
    params: Params,
    renderer: MolThumbnailer,
    thumb_size: u32,
    next_to_render: usize,
    folder: PathBuf,
}

impl MolBrowserApp {
    fn new(folder: PathBuf, params: Params, renderer: MolThumbnailer, thumb_size: u32) -> Self {
        let mut entries = Vec::new();
        println!("DEBUG: folder path = {:?}", folder);
        println!("DEBUG: folder.exists() = {}", folder.exists());
        println!("DEBUG: folder.is_dir() = {}", folder.is_dir());
        if folder.is_dir() {
            let dir_iter = match std::fs::read_dir(&folder) {
                Ok(it) => it,
                Err(e) => {
                    println!("ERROR: cannot read_dir {:?}: {}", folder, e);
                    return Self { entries, params, renderer, thumb_size, next_to_render: 0, folder };
                }
            };
            let mut paths: Vec<PathBuf> = dir_iter
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |ext| ext == "xyz"))
                .collect();
            paths.sort();
            println!("DEBUG: found {} .xyz files", paths.len());
            for path in &paths {
                println!("DEBUG:   {:?}", path);
            }
            for path in paths {
                let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
                let natoms = read_xyz(&path).map(|s| s.apos.len()).unwrap_or(0);
                entries.push(MolEntry { path, name, natoms, thumbnail: None, texture: None });
            }
        } else {
            println!("ERROR: folder {:?} is not a directory", folder);
        }
        println!("DEBUG: created {} entries", entries.len());
        Self { entries, params, renderer, thumb_size, next_to_render: 0, folder }
    }

    fn render_next(&mut self) {
        if self.next_to_render >= self.entries.len() { return; }
        let idx = self.next_to_render;
        self.next_to_render += 1;

        let entry = &mut self.entries[idx];
        let xyz = match read_xyz(&entry.path) {
            Ok(x) => x,
            Err(_) => return,
        };
        if xyz.apos.is_empty() { return; }

        // Build bonds by covalent radii, keeping at most 4 closest per atom
        let radii: Vec<f64> = xyz.elems.iter().map(|el| {
            self.params.get_element_type(el).map(|et| et.r_cov).unwrap_or(1.0)
        }).collect();
        let mut bonds: Vec<[usize; 2]> = Vec::new();
        let n = xyz.apos.len();
        for i in 0..n {
            let mut candidates: Vec<(usize, f64)> = Vec::new();
            for j in (i+1)..n {
                let d = surfmol_common::math::vec3::Vec3d::set_sub(xyz.apos[j], xyz.apos[i]);
                let rcut = radii[i] + radii[j] + 0.4;
                let dist2 = d.norm2();
                if dist2 < rcut * rcut {
                    candidates.push((j, dist2));
                }
            }
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            for (j, _) in candidates.iter().take(4) {
                bonds.push([i, *j]);
            }
        }

        let rgba = self.renderer.render(self.thumb_size, &xyz.apos, &xyz.elems, &bonds, &self.params);
        let img = egui::ColorImage::from_rgba_unmultiplied(
            [self.thumb_size as usize, self.thumb_size as usize],
            &rgba,
        );
        entry.thumbnail = Some(img);
    }
}

impl eframe::App for MolBrowserApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Render up to 4 thumbnails per frame to keep UI responsive
        for _ in 0..4 {
            self.render_next();
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(16));

        // Upload any new thumbnails to GPU textures
        for entry in &mut self.entries {
            if let Some(img) = &entry.thumbnail {
                if entry.texture.is_none() {
                    let tex = ctx.load_texture(&entry.name, img.clone(), egui::TextureOptions::LINEAR);
                    entry.texture = Some(tex);
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("MolBrowser");
                ui.label(format!("Folder: {}", self.folder.display()));
                ui.label(format!("Files: {}", self.entries.len()));
                ui.label(format!("Rendered: {} / {}", self.next_to_render, self.entries.len()));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let thumb_w = self.thumb_size as f32;
                let spacing = 8.0;
                let available_w = ui.available_width();
                let cols = ((available_w + spacing) / (thumb_w + spacing)).floor() as usize;
                let cols = cols.max(1);

                for chunk in self.entries.chunks(cols) {
                    ui.horizontal(|ui| {
                        for entry in chunk {
                            ui.vertical(|ui| {
                                let size = egui::vec2(thumb_w, thumb_w);
                                if let Some(texture) = &entry.texture {
                                    ui.image(texture);
                                } else {
                                    ui.allocate_space(size);
                                }
                                ui.label(&entry.name);
                                ui.label(format!("{} atoms", entry.natoms));
                            });
                            ui.add_space(spacing);
                        }
                    });
                    ui.add_space(spacing);
                }
            });
        });
    }
}

// ------------------------------------------------------------------
// Main
// ------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let workspace_root = PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("../..");
    let folder = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        // Try to find data/xyz relative to executable or workspace
        let exe = std::env::current_exe().unwrap();
        let mut p = exe.parent().unwrap().to_path_buf();
        for _ in 0..4 {
            let candidate = p.join("data/xyz");
            if candidate.exists() { return candidate; }
            if !p.pop() { break; }
        }
        PathBuf::from("data/xyz")
    });
    // If provided path doesn't exist, try relative to workspace root
    let folder = if folder.exists() {
        folder
    } else {
        let candidate = workspace_root.join(&folder);
        if candidate.exists() {
            println!("DEBUG: resolved folder to workspace root: {:?}", candidate);
            candidate
        } else {
            println!("DEBUG: folder {:?} not found; using as-is", folder);
            folder
        }
    };

    // Load params
    let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.join("../..");
    let dat_dir = if folder.join("../AtomTypes.dat").exists() {
        folder.join("..")
    } else if workspace_root.join("data/AtomTypes.dat").exists() {
        workspace_root.join("data")
    } else {
        // Try workspace root from executable
        let mut p = std::env::current_exe().unwrap();
        for _ in 0..5 {
            let candidate = p.join("data");
            if candidate.join("AtomTypes.dat").exists() { break; }
            if !p.pop() { break; }
        }
        p.join("data")
    };

    let mut params = Params::new();
    if dat_dir.join("ElementTypes.dat").exists() {
        params.load_element_types(dat_dir.join("ElementTypes.dat"));
        params.load_atom_types(dat_dir.join("AtomTypes.dat"));
        params.load_bond_types(dat_dir.join("BondTypes.dat"));
        params.load_angle_types(dat_dir.join("AngleTypes.dat"));
        if dat_dir.join("DihedralTypes.dat").exists() {
            params.load_dihedral_types(dat_dir.join("DihedralTypes.dat"));
        }
        println!("Loaded {} elements, {} atom types", params.elements.len(), params.atom_types.len());
    } else {
        println!("WARNING: .dat files not found in {:?}", dat_dir);
    }

    let renderer = MolThumbnailer::new();

    let app = MolBrowserApp::new(folder, params, renderer, 128);

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "MolBrowser",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    ).unwrap();
}
