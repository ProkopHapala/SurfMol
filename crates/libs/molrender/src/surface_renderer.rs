use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::impostor::CameraData;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadVertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

const SURFACE_SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec3<f32>, _pad1: f32,
    right: vec3<f32>, _pad2: f32,
    up: vec3<f32>, _pad3: f32,
    forward: vec3<f32>, ortho: f32,
    ray_shift: vec4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

struct VertexInput {
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = cam.view_proj * vec4<f32>(in.pos, 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
"#;

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 1, 3, 2];

pub struct SurfaceRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::RenderPipeline,
    camera_buf: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    ib: wgpu::Buffer,
}

impl SurfaceRenderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("surface_shader"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface_bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("surface_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("surface_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<QuadVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
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

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface_camera"),
            size: std::mem::size_of::<CameraData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("surface_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface_ib"),
            contents: bytemuck::cast_slice(&QUAD_INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self { device, queue, pipeline, camera_buf, bind_group_layout, sampler, ib }
    }

    /// Render a parallelogram textured quad defined by origin + u_edge * s + v_edge * t.
    /// origin, u_edge, v_edge are in world space; z component of origin is the surface plane.
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, target_view: &wgpu::TextureView, depth_view: &wgpu::TextureView, camera: &CameraData, texture_view: &wgpu::TextureView, origin: [f32; 3], u_edge: [f32; 3], v_edge: [f32; 3]) {
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));

        let verts = [
            QuadVertex { pos: origin,                                    uv: [0.0, 0.0] },
            QuadVertex { pos: [origin[0]+u_edge[0], origin[1]+u_edge[1], origin[2]+u_edge[2]], uv: [1.0, 0.0] },
            QuadVertex { pos: [origin[0]+v_edge[0], origin[1]+v_edge[1], origin[2]+v_edge[2]], uv: [0.0, 1.0] },
            QuadVertex { pos: [origin[0]+u_edge[0]+v_edge[0], origin[1]+u_edge[1]+v_edge[1], origin[2]+u_edge[2]+v_edge[2]], uv: [1.0, 1.0] },
        ];
        let vb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("surface_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(texture_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(self.ib.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..6, 0, 0..1);
        }
    }
}
