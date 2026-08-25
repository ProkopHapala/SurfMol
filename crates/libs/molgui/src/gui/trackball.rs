use glam::{Quat, Vec2, Vec3};
use molrender::impostor::CameraData;

pub struct TrackballCam {
    pub target: Vec3,
    pub rotation: Quat,
    pub target_rotation: Quat,
    pub dist_cam: f32,
    pub zoom: f32,
    pub lerp_speed: f32,
}

impl TrackballCam {
    pub fn new(target: Vec3, dist: f32) -> Self {
        Self { target, rotation: Quat::IDENTITY, target_rotation: Quat::IDENTITY, dist_cam: dist, zoom: dist, lerp_speed: 20.0 }
    }
    pub fn pos(&self) -> Vec3 { self.target + self.rotation * Vec3::new(0.0, 0.0, self.dist_cam) }
    pub fn fwd(&self) -> Vec3 { (self.target - self.pos()).normalize() }
    pub fn up(&self) -> Vec3 { (self.rotation * Vec3::new(0.0, 1.0, 0.0)).normalize() }
    pub fn right(&self) -> Vec3 { self.fwd().cross(self.up()).normalize() }
    pub fn zoom(&mut self, delta: f32) { self.zoom *= 1.0 + delta * 0.1; self.zoom = self.zoom.clamp(0.2, 200.0); }
    pub fn pan(&mut self, dx: f32, dy: f32) { let s = self.zoom * 0.001; self.target += self.right() * (-dx * s) + self.up() * (dy * s); }
    pub fn mouse_to_sphere(&self, mouse: Vec2, w: f32, h: f32) -> Vec3 {
        let radius = w.min(h) * 0.4; let x = (mouse.x - w * 0.5) / radius; let y = -(mouse.y - h * 0.5) / radius; let r2 = x * x + y * y;
        if r2 <= 1.0 { Vec3::new(x, y, (1.0f32 - r2).sqrt()) } else { let s = 1.0f32 / r2.sqrt(); Vec3::new(x * s, y * s, 0.0) }
    }
    pub fn rotate(&mut self, prev: Vec2, curr: Vec2, w: f32, h: f32) {
        let a = self.mouse_to_sphere(prev, w, h); let b = self.mouse_to_sphere(curr, w, h);
        let q = Quat::from_rotation_arc(b, a); self.target_rotation = (self.target_rotation * q).normalize();
    }
    pub fn update(&mut self, dt: f32) {
        let t = (self.lerp_speed * dt).clamp(0.0, 1.0);
        self.rotation = self.rotation.slerp(self.target_rotation, t);
    }
    pub fn screen_ray(&self, mouse: Vec2, w: f32, h: f32) -> (Vec3, Vec3) {
        let fwd = self.fwd(); let right = self.right(); let up = self.up();
        let mx = mouse.x; let my = h - mouse.y;
        let mbx = (2.0 * mx - w) * self.zoom / h; let mby = (2.0 * my - h) * self.zoom / h;
        (self.pos() + right * mbx + up * mby, fwd)
    }
    pub fn camera_data(&self, w: u32, h: u32) -> CameraData {
        let eye_v = self.pos(); let target_v = self.target; let _up_v = self.up();
        let eye = [eye_v.x, eye_v.y, eye_v.z];
        let aspect = (w as f32 / h as f32).max(1e-6); let zoom = self.zoom;
        let r = self.right(); let u = self.up(); let f = (eye_v - target_v).normalize();
        let sx = 1.0 / (zoom * aspect);
        let sy = 1.0 / zoom;
        let sz = -1.0 / (1000.0 - 0.01);
        let tz = -0.01 / (1000.0 - 0.01);
        let tx = -(r.x * eye[0] + r.y * eye[1] + r.z * eye[2]);
        let ty = -(u.x * eye[0] + u.y * eye[1] + u.z * eye[2]);
        let tz_view = -(f.x * eye[0] + f.y * eye[1] + f.z * eye[2]);
        let vp = [
            [sx * r.x, sx * r.y, sx * r.z, sx * tx],
            [sy * u.x, sy * u.y, sy * u.z, sy * ty],
            [sz * f.x, sz * f.y, sz * f.z, sz * tz_view + tz],
            [0.0, 0.0, 0.0, 1.0],
        ];
        CameraData { view_proj: vp, eye, _pad1: 0.0, right: [r.x, r.y, r.z], _pad2: 0.0, up: [u.x, u.y, u.z], _pad3: 0.0, forward: [f.x, f.y, f.z], ortho: 1.0, ray_shift: [10000.0, 0.0, 0.0, 0.0] }
    }
}
