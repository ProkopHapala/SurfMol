// raff3d_mols — 3 pre-formed methane-like clusters (sp3 + 4 h on ~1.55 A
// tetrahedral dirs, small jitter) + 6 random sp3 extras, soft box, Grid3 PIC
// neighbor search. Select/drag an atom to see its cell + forward stencil
// (amber) and rcut interaction lines (cyan=bonding, red=repulsive).
// Tetra dirs (±1,±1,±1)/sqrt3 * 1.55 ≈ ±0.9 per axis.

// cluster A at (-4,-1,-1)
atom(-4.0, -1.0, -1.0, "sp3");
atom(-3.10, -0.10, -0.10, "h");
atom(-3.10, -1.90, -1.90, "h");
atom(-4.90, -0.10, -1.90, "h");
atom(-4.90, -1.90, -0.10, "h");

// cluster B at (0,1,1)
atom( 0.0,  1.0,  1.0, "sp3");
atom( 0.92,  1.91,  1.88, "h");
atom( 0.88,  0.12,  0.09, "h");
atom(-0.91,  1.88,  0.11, "h");
atom(-0.89,  0.09,  1.92, "h");

// cluster C at (4,-1,1)
atom( 4.0, -1.0,  1.0, "sp3");
atom( 4.91, -0.09,  1.90, "h");
atom( 4.89, -1.91,  0.10, "h");
atom( 3.09, -0.11,  0.12, "h");
atom( 3.11, -1.90,  1.91, "h");

random(6, 11, 2.0, "sp3");          // extras wandering between clusters
box(-8.0, -6.0, -6.0, 8.0, 6.0, 6.0, 15.0);
grid(2.0);
field(false);
