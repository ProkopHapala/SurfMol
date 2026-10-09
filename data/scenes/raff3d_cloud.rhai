// raff3d_cloud — 12 sp3 C + 12 H uniform-random in a sphere, soft box walls,
// Grid3 neighbor search. Watch bonds emerge and clusters assemble.
random(12, 7, 2.5, "sp3");
random(12, 8, 2.5, "h");
box(-4.0, -4.0, -4.0, 4.0, 4.0, 4.0, 20.0);
grid(2.0);
field(true);
