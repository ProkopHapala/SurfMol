pub mod impostor;
pub mod line_renderer;
pub mod surface_renderer;

use std::sync::Arc;
use surfmol_common::math::vec3::Vec3d;
use surfmol_topology::params::Params;

// ------------------------------------------------------------------
// ThumbnailRenderer (wraps ImpostorRenderer for offscreen use)
// ------------------------------------------------------------------

pub struct ThumbnailRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    impostor: impostor::ImpostorRenderer,
}

impl ThumbnailRenderer {
    pub fn new(_size: u32) -> Self {
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
                label: Some("render_device"),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                trace: wgpu::Trace::Off,
            },
        )).expect("no device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let impostor = impostor::ImpostorRenderer::new(device.clone(), queue.clone(), 10000, wgpu::TextureFormat::Rgba8UnormSrgb);
        Self { device, queue, impostor }
    }

    pub fn render(&mut self, size: u32, apos: &[Vec3d], elems: &[String], _bonds: &[[usize; 2]], params: &Params) -> Vec<u8> {
        if apos.is_empty() {
            return vec![0u8; (size * size * 4) as usize];
        }

        // Build atom instances
        let mut instances = Vec::with_capacity(apos.len());
        for (i, p) in apos.iter().enumerate() {
            let el = elems.get(i).map(|s| s.as_str()).unwrap_or("C");
            let col = element_color_f32(el, params);
            let r = params.get_element_type(el).map(|e| e.r_vdw as f32).unwrap_or(1.7) * 0.3;
            instances.push(impostor::AtomInstance {
                pos: [p.x as f32, p.y as f32, p.z as f32],
                radius: r,
                color: col,
                _pad: 0.0,
            });
        }
        self.impostor.set_atoms(&instances);
        self.impostor.set_target_size(size, size);

        // Camera
        let (mut mn, mut mx) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in apos {
            let a = [p.x as f32, p.y as f32, p.z as f32];
            for i in 0..3 { mn[i] = mn[i].min(a[i]); mx[i] = mx[i].max(a[i]); }
        }
        let center = impostor::mul3s(impostor::add3(mn, mx), 0.5);
        let span = impostor::sub3(mx, mn);
        let max_span = span[0].max(span[1]).max(span[2]).max(2.0);
        let rmax = 3.0f32;
        let ortho_half = max_span * 0.5 + rmax;

        let proj = impostor::ortho(-ortho_half, ortho_half, -ortho_half, ortho_half, 1.0, 1000.0);
        let eye = impostor::add3(center, [ortho_half * 2.0, ortho_half * 1.5, ortho_half * 2.5]);
        let view = impostor::look_at(eye, center, [0.0, 1.0, 0.0]);
        let vp = impostor::mul4x4(view, proj);

        let z = impostor::normalize3(impostor::sub3(eye, center));
        let x = impostor::normalize3(impostor::cross3([0.0, 1.0, 0.0], z));
        let y = impostor::cross3(z, x);
        let fwd = impostor::normalize3(impostor::sub3(center, eye));

        let camera = impostor::CameraData {
            view_proj: impostor::transpose4x4(vp),
            eye,
            _pad1: 0.0,
            right: x,
            _pad2: 0.0,
            up: y,
            _pad3: 0.0,
            forward: fwd,
            ortho: 0.0,
            ray_shift: [0.0, 0.0, 0.0, 0.0],
        };

        // Offscreen color texture
        let out_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("out_tex"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.impostor.render(&out_view, wgpu::Color { r: 0.08, g: 0.08, b: 0.12, a: 1.0 }, &camera);

        // Readback
        let buf_size = (size * size * 4) as wgpu::BufferAddress;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"), size: buf_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("thumb_encoder") });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &out_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size * 4), rows_per_image: Some(size) } },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity(data.len());
        rgba.extend_from_slice(&data);
        drop(data);
        readback.unmap();
        rgba
    }
}

fn element_color_f32(elem: &str, params: &Params) -> [f32; 3] {
    params.get_element_type(elem)
        .map(|et| {
            let c = et.color;
            [((c >> 16) & 0xFF) as f32 / 255.0, ((c >> 8) & 0xFF) as f32 / 255.0, (c & 0xFF) as f32 / 255.0]
        })
        .unwrap_or([0.78, 0.78, 0.78])
}
