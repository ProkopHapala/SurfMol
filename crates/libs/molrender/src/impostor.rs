use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;


// ------------------------------------------------------------------
// GPU data layouts (must match WGSL structs exactly)
// ------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct AtomInstance {
    pub pos: [f32; 3],      // 0
    pub radius: f32,        // 12
    pub color: [f32; 3],    // 16
    pub _pad: f32,          // 28  (keeps struct at 32 bytes, vec4-aligned)
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraData {
    pub view_proj: [[f32; 4]; 4], // 0
    pub eye: [f32; 3],             // 64
    pub _pad1: f32,
    pub right: [f32; 3],           // 80
    pub _pad2: f32,
    pub up: [f32; 3],              // 96
    pub _pad3: f32,
    pub forward: [f32; 3],         // 112
    pub ortho: f32,                // 124  1.0 => orthographic ray mode
    pub ray_shift: [f32; 4],       // 128  ray_shift[0] = shift ray origin along -forward (world units)
}

// ------------------------------------------------------------------
// Shaders
// ------------------------------------------------------------------

const IMPOSTOR_SHADER: &str = r"
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec3<f32>, _pad1: f32,
    right: vec3<f32>, _pad2: f32,
    up: vec3<f32>, _pad3: f32,
    forward: vec3<f32>, ortho: f32,
    ray_shift: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct Atom {
    pos: vec3<f32>, radius: f32,
    color: vec3<f32>, _pad: f32,
};
@group(0) @binding(1) var<storage, read> atoms: array<Atom>;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) world: vec3<f32>,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@location(0) q: vec2<f32>, @builtin(instance_index) ii: u32) -> VertexOut {
    let a = atoms[ii];
    let w = a.pos + cam.right * a.radius * q.x + cam.up * a.radius * q.y;
    return VertexOut(
        cam.view_proj * vec4<f32>(w, 1.0),
        a.color,
        a.pos,
        a.radius,
        w
    );
}

@fragment
fn fs_main(v: VertexOut) -> FragOut {
    let ray_dir = -cam.forward;
    let shift = v.radius + 1.0;
    let ray_origin = v.world + cam.forward * shift;
    let oc = ray_origin - v.center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - v.radius * v.radius;
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) { discard; }
    let t = (-b - sqrt(disc)) / (2.0 * a);
    if (t < 0.0) { discard; }
    let hit = ray_origin + ray_dir * t;
    let n = normalize(hit - v.center);
    let light = normalize(vec3<f32>(0.3, 0.5, 0.8));
    let ndotl = max(dot(n, light), 0.0);
    let col = v.color * (0.3 + 0.7 * ndotl);
    let clip = cam.view_proj * vec4<f32>(hit, 1.0);
    return FragOut(clip.z / clip.w, vec4<f32>(col, 1.0));
}
";

// ------------------------------------------------------------------
// Renderer
// ------------------------------------------------------------------

pub struct ImpostorRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,

    // One billboard quad (vertex-stepped)
    quad_vb: wgpu::Buffer,
    quad_ib: wgpu::Buffer,
    num_indices: u32,

    // Atom storage buffer (capacity >= max_atoms, updated via write_buffer)
    atom_buf: wgpu::Buffer,
    atom_count: usize,
    max_atoms: usize,

    // Camera uniform
    camera_buf: wgpu::Buffer,

    // Depth target (resized by set_target_size)
    depth_tex: wgpu::Texture,
    depth_view: wgpu::TextureView,
}

impl ImpostorRenderer {
    /// Create a new renderer. `max_atoms` reserves GPU buffer capacity.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, max_atoms: usize, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("impostor_shader"),
            source: wgpu::ShaderSource::Wgsl(IMPOSTOR_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("impostor_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("impostor_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("impostor_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 2]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // billboards must never be culled
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Billboard quad: 4 corners, 2 triangles
        let quad_verts: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]];
        let quad_indices: [u16; 6] = [0, 1, 2, 1, 3, 2];
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("impostor_quad_vb"),
            contents: bytemuck::cast_slice(&quad_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("impostor_quad_ib"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let atom_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("impostor_atoms"),
            size: (max_atoms * std::mem::size_of::<AtomInstance>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("impostor_camera"),
            size: std::mem::size_of::<CameraData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (depth_tex, depth_view) = Self::make_depth(&device, 1, 1);

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            quad_vb,
            quad_ib,
            num_indices: quad_indices.len() as u32,
            atom_buf,
            atom_count: 0,
            max_atoms,
            camera_buf,
            depth_tex,
            depth_view,
        }
    }

    fn make_depth(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("impostor_depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }

    /// Access depth view for subsequent render passes (e.g. lines).
    pub fn depth_view(&self) -> &wgpu::TextureView { &self.depth_view }

    #[inline(always)]
    pub fn max_atoms(&self) -> usize { self.max_atoms }

    /// Resize depth buffer to match target dimensions.
    pub fn set_target_size(&mut self, width: u32, height: u32) {
        if self.depth_tex.width() != width || self.depth_tex.height() != height {
            let (tex, view) = Self::make_depth(&self.device, width, height);
            self.depth_tex = tex;
            self.depth_view = view;
        }
    }

    /// Upload new atom data. `atoms.len()` must be <= `max_atoms`.
    pub fn set_atoms(&mut self, atoms: &[AtomInstance]) {
        assert!(atoms.len() <= self.max_atoms, "atom count {} exceeds capacity {}", atoms.len(), self.max_atoms);
        self.atom_count = atoms.len();
        if !atoms.is_empty() {
            self.queue.write_buffer(&self.atom_buf, 0, bytemuck::cast_slice(atoms));
        }
    }

    /// Upload a complete fresh atom array in one GPU transfer.
    /// Call this each frame when positions change during MD.
    pub fn set_instances(&mut self, instances: &[AtomInstance]) {
        self.set_atoms(instances);
    }

    /// Build bind group for a draw call.
    fn make_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("impostor_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.atom_buf.as_entire_binding() },
            ],
        })
    }

    /// Render all atoms into `target_view`.
    pub fn render(&self, target_view: &wgpu::TextureView, clear: wgpu::Color, camera: &CameraData) {
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(&[*camera]));
        let bind_group = self.make_bind_group();

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("impostor_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("impostor_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            if self.atom_count > 0 {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_vertex_buffer(0, self.quad_vb.slice(..));
                pass.set_index_buffer(self.quad_ib.slice(..), wgpu::IndexFormat::Uint16);
                pass.draw_indexed(0..self.num_indices, 0, 0..self.atom_count as u32);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
