// naphthalene — two fused hexagons + 2 ring entities + 4 loose atoms to play with.
// Ring entities pin site attraction on the circumference + repulsive core in the hole.

ring(-1.1258, 0.0, 6, 1.3);   // left ring entity (x,y,nfold,R)
ring( 1.1258, 0.0, 6, 1.3);   // right ring entity

// hexagon vertices of both rings (shared edge at x=0, +/-0.65)
atom(-2.2517,  0.65,   30.0);
atom(-2.2517, -0.65,  -30.0);
atom(-1.1258, -1.30,  -90.0);
atom( 0.0,    -0.65, -150.0);   // shared edge (fused)
atom( 0.0,     0.65,  150.0);   // shared edge (fused)
atom(-1.1258,  1.30,   90.0);
atom( 1.1258,  1.30,   90.0);
atom( 1.1258, -1.30,  -90.0);
atom( 2.2517, -0.65,  -150.0);
atom( 2.2517,  0.65,   150.0);

// loose atoms — drag them onto free sites or into a ring hole
atom( 4.5,  1.5,  45.0);
atom(-4.0, -2.0, 210.0);
atom( 3.8, -2.5,   0.0);
atom(-3.5,  2.8, 300.0);

arena(7.0, 2.0);
// grid(1.0);
