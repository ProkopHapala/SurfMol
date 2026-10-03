// pair_probe — mutual-gate sandbox: ONE pinned atom at center, two free atoms,
// NO ring entities. Drag a free atom around the pinned one and watch the
// probe-field: attraction basins exist only where BOTH atoms can face each other.
// Rhai API: atom(x,y,phi_deg) / pinned(x,y,phi_deg) / ring / arena / grid / w,a,b

pinned(0.0, 0.0, 0.0);        // immovable central atom, ports at 0/120/240 deg
atom( 2.2, 0.3, 180.0);       // free atom A — facing the center
atom(-1.8, -1.4, 0.0);        // free atom B — random orientation

arena(6.0, 2.0);              // soft confinement (optional; comment out for open space)
grid(6.0);                    // CSR grid dx = rcut -> 3x3 stencil (uncheck 'use CSR grid' for O(N^2))
// w(0.45);                   // gate width override (default 0.3)
