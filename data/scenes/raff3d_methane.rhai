// raff3d_methane — C at origin + 4 H on a 1.6 A tetrahedral sphere (small jitter).
// Should self-assemble 4 C-H bonds (E ~ -4 eV). Space=run, R=reload.
atom(0.0, 0.0, 0.0, "sp3");
atom( 1.0338,  0.8738,  0.9938, "h");   // (1,1,1)/sqrt3 * 1.6 + jitter
atom( 0.8538, -0.8138, -0.9538, "h");   // (1,-1,-1)/sqrt3 * 1.6 + jitter
atom(-0.8838,  1.0338, -0.8638, "h");   // (-1,1,-1)/sqrt3 * 1.6 + jitter
atom(-1.0138, -0.9438,  1.0238, "h");   // (-1,-1,1)/sqrt3 * 1.6 + jitter
field(true);
