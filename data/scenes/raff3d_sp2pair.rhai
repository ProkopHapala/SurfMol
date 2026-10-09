// raff3d_sp2pair — 2 sp2 atoms 2.5 A apart, one pinned. Identity quats keep
// the 3 in-plane ports in z=0 so the field slice (slice_z=0) shows the
// trigonal lobes — contrast with sp3 (raff3d_pair) where ports point out of
// plane. Drag the free atom with LMB.
atom(-1.25, 0.0, 0.0, "sp2");
pinned(1.25, 0.0, 0.0, "sp2");
field(true);
