use std::sync::Arc;
use numcore::math::vec3::Vec3d;
use moltopo::params::Params;
use molrender::impostor::{ImpostorRenderer, AtomInstance, CameraData};
use molrender::line_renderer::{LineRenderer, LineVertex};
use molrender::impostor::{sub3, normalize3};
 
/// External harness for molecule thumbnail generation.
/// Uses surfmol-molrender GPU primitives but handles alignment,
/// ortho camera fitting, and bond rendering externally.
pub struct MolThumbnailer {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) impostor: ImpostorRenderer,
    pub(crate) lines: LineRenderer,
}
 
impl MolThumbnailer {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })).expect("no adapter");
 
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: Some("thumb_device"),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            },
        )).expect("no device");
 
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let impostor = ImpostorRenderer::new(device.clone(), queue.clone(), 10000, fmt);
        let lines = LineRenderer::new(device.clone(), queue.clone(), fmt);
        Self { device, queue, impostor, lines }
    }
 
    pub fn render(&mut self, size: u32, apos: &[Vec3d], elems: &[String], bonds: &[[usize; 2]], params: &Params) -> Vec<u8> {
        if apos.is_empty() {
            return vec![0u8; (size * size * 4) as usize];
        }
 
        // 1. Align to principal axes (longest in x, y; shortest in z)
        let aligned = align_to_principal_axes(apos);
 
        // 2. Build atom instances
        let mut instances = Vec::with_capacity(aligned.len());
        let mut rmax = 0.0f32;
        for (i, p) in aligned.iter().enumerate() {
            let el = elems.get(i).map(|s| s.as_str()).unwrap_or("C");
            let col = params.element_color_f32(el);
            let r = params.get_element_type(el).map(|e| e.r_vdw as f32).unwrap_or(1.7) * 0.3;
            if r > rmax { rmax = r; }
            instances.push(AtomInstance { pos: *p, radius: r, color: col, _pad: 0.0 });
        }
        self.impostor.set_atoms(&instances);
        self.impostor.set_target_size(size, size);
 
        // 3. Fit ortho camera to aligned bounds
        let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in &aligned {
            for i in 0..3 { mn[i] = mn[i].min(p[i]); mx[i] = mx[i].max(p[i]); }
        }
        let center = [(mn[0]+mx[0])*0.5, (mn[1]+mx[1])*0.5, (mn[2]+mx[2])*0.5];
        let wx = mx[0] - mn[0];
        let wy = mx[1] - mn[1];
        let wz = mx[2] - mn[2];
        let half_x = wx * 0.5 + rmax;
        let half_y = wy * 0.5 + rmax;
        let half = half_x.max(half_y); // preserve aspect ratio
        let half_z = (wz * 0.5 + rmax).max(1.0);
 
        let zoom = half;
        let aspect = 1.0f32;
        let near = 0.01f32;
        let far = 1000.0f32;
        let dist = half_z * 4.0;
        let eye = [center[0], center[1], center[2] + dist];
        let right = [1.0f32, 0.0, 0.0];
        let up = [0.0f32, 1.0, 0.0];
        let f = normalize3(sub3(eye, center));
 
        let sx = 1.0 / (zoom * aspect);
        let sy = 1.0 / zoom;
        let sz = -1.0 / (far - near);
        let tz = -near / (far - near);
        let tx = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
        let ty = -(up[0] * eye[0] + up[1] * eye[1] + up[2] * eye[2]);
        let tz_view = -(f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]);
        let vp = [
            [sx * right[0], sx * right[1], sx * right[2], sx * tx],
            [sy * up[0], sy * up[1], sy * up[2], sy * ty],
            [sz * f[0], sz * f[1], sz * f[2], sz * tz_view + tz],
            [0.0, 0.0, 0.0, 1.0],
        ];
 
        let camera = CameraData {
            view_proj: vp,
            eye,
            _pad1: 0.0,
            right,
            _pad2: 0.0,
            up,
            _pad3: 0.0,
            forward: f,
            ortho: 1.0,
            ray_shift: [0.0, 0.0, 0.0, 0.0],
        };
 
        // 4. Offscreen target
        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("thumb_out"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());
 
        // 5. Render atoms
        self.impostor.render(&out_view, wgpu::Color { r: 0.08, g: 0.08, b: 0.12, a: 1.0 }, &camera);
 
        // 6. Render bonds
        if !bonds.is_empty() {
            let mut verts = Vec::with_capacity(bonds.len() * 2);
            let col = [0.5f32, 0.5, 0.5, 1.0];
            for b in bonds {
                let i = b[0]; let j = b[1];
                if i >= aligned.len() || j >= aligned.len() { continue; }
                verts.push(LineVertex { pos: aligned[i], col });
                verts.push(LineVertex { pos: aligned[j], col });
            }
            if verts.len() >= 2 {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("bond_enc") });
                self.lines.render(&mut enc, &out_view, self.impostor.depth_view(), &camera, &verts);
                self.queue.submit(std::iter::once(enc.finish()));
            }
        }
 
        // 7. Readback
        let buf_size = (size * size * 4) as wgpu::BufferAddress;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("thumb_rb"), size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy_enc") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &out_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * 4), rows_per_image: Some(size) } },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(enc.finish()));
 
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity(data.len());
        rgba.extend_from_slice(&data);
        drop(data);
        readback.unmap();
        rgba
    }
}
 
pub(crate) fn align_to_principal_axes(apos: &[Vec3d]) -> Vec<[f32; 3]> {
    let n = apos.len() as f64;
    let cog = Vec3d::new(
        apos.iter().map(|p| p.x).sum::<f64>() / n,
        apos.iter().map(|p| p.y).sum::<f64>() / n,
        apos.iter().map(|p| p.z).sum::<f64>() / n,
    );
 
    let mut ixx = 0.0f64; let mut iyy = 0.0f64; let mut izz = 0.0f64;
    let mut ixy = 0.0f64; let mut ixz = 0.0f64; let mut iyz = 0.0f64;
    for p in apos {
        let x = p.x - cog.x;
        let y = p.y - cog.y;
        let z = p.z - cog.z;
        ixx += y*y + z*z;
        iyy += x*x + z*z;
        izz += x*x + y*y;
        ixy -= x*y;
        ixz -= x*z;
        iyz -= y*z;
    }
 
    let mat = [
        ixx as f32, ixy as f32, ixz as f32,
        ixy as f32, iyy as f32, iyz as f32,
        ixz as f32, iyz as f32, izz as f32,
    ];
    let eig = numcore::math::linalg::symmetric_eigen_3x3(mat);

    // smallest eigenvalue => largest spatial extent => x (longest)
    // largest eigenvalue  => smallest spatial extent => z (view axis)
    // eig is already sorted ascending by eigenvalue
    let ex = &eig[0].1;
    let ey = &eig[1].1;
    let ez = &eig[2].1;

    let mut out = Vec::with_capacity(apos.len());
    for p in apos {
        let vx = (p.x - cog.x) as f32;
        let vy = (p.y - cog.y) as f32;
        let vz = (p.z - cog.z) as f32;
        out.push([
            ex[0] * vx + ex[1] * vy + ex[2] * vz,
            ey[0] * vx + ey[1] * vy + ey[2] * vz,
            ez[0] * vx + ez[1] * vy + ez[2] * vz,
        ]);
    }
    out
}
 