use molrender::impostor::{ImpostorRenderer, AtomInstance, CameraData, look_at, ortho, mul4x4, normalize3, sub3, cross3};

#[test]
fn impostor_single_atom() {
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
            label: Some("test_device"),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        },
    )).expect("no device");

    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    let mut renderer = ImpostorRenderer::new(device.clone(), queue.clone(), 1024, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer.set_target_size(128, 128);

    let atoms = vec![AtomInstance {
        pos: [0.0, 0.0, 0.0],
        radius: 1.0,
        color: [1.0, 0.0, 0.0],
        _pad: 0.0,
    }];
    renderer.set_atoms(&atoms);

    let eye = [0.0, 0.0, 5.0];
    let target = [0.0, 0.0, 0.0];
    let up = [0.0, 1.0, 0.0];
    let view = look_at(eye, target, up);
    let proj = ortho(-3.0, 3.0, -3.0, 3.0, 0.1, 100.0);
    // Row-major row-vector M (clip = point * M); upload directly — WGSL column-major
    // byte interpretation transposes it, so M_wgsl = M^T and M_wgsl * v = v * M.
    let vp = mul4x4(view, proj);

    let z = normalize3(sub3(eye, target));
    let right = normalize3(cross3([0.0, 1.0, 0.0], z));
    let upv = cross3(z, right);

    let cam = CameraData {
        view_proj: vp,
        eye,
        _pad1: 0.0,
        right,
        _pad2: 0.0,
        up: upv,
        _pad3: 0.0,
        forward: normalize3(sub3(target, eye)),
        ortho: 0.0,
        ray_shift: [0.0, 0.0, 0.0, 0.0],
    };

    let out_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_out"),
        size: wgpu::Extent3d { width: 128, height: 128, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let out_view = out_tex.create_view(&wgpu::TextureViewDescriptor::default());

    renderer.render(&out_view, wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }, &cam);

    let buf_size = (128 * 128 * 4) as wgpu::BufferAddress;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback_enc") });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &out_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(128 * 4), rows_per_image: Some(128) } },
        wgpu::Extent3d { width: 128, height: 128, depth_or_array_layers: 1 },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    let data = slice.get_mapped_range();
    let mut non_bg = 0;
    for i in (0..data.len()).step_by(4) {
        if data[i] > 10 || data[i+1] > 10 || data[i+2] > 10 {
            non_bg += 1;
        }
    }
    println!("impostor single atom non-bg pixels: {} / {}", non_bg, data.len()/4);
    drop(data);
    readback.unmap();

    assert!(non_bg > 0, "impostor single atom at origin should produce visible pixels");
}
