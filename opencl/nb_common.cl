// nb_common.cl — Non-bonded loop macro fragments (Axis 1 + Axis 2).
//
// Provides the two macro axes that vary independently in the generic
// getNonBonded template:
//
//   Axis 1 — NB_PAIR_FORCE(dp, REQK, R2damp)  : pairwise potential
//   Axis 2 — NB_EXCL_ARGS / NB_EXCL_SETUP / NB_EXCL_TEST / NB_EXCL_PBC_TEST
//                                            : exclusion strategy
//
// Each axis has multiple variants. The ClAssembler substitutes the chosen
// variant via Substitutions before compilation.
//
// Requires common.cl + Forces.cl concatenated first (for getLJQH,
// mixREQ_arithmetic, float4Zero, cl_Mat3, etc.).
//
// Reference: FireCore cpp/common_resources/cl/UFF.cl:getNonBond (neighs4)
//            and getNonBond_ex2 (packed exclusion list).

// ======================================================================
//  Axis 1 — Pairwise potential macros
// ======================================================================

// NB_PAIR_LJQH — Lennard-Jones + damped Coulomb + H-bond (UFF + SPFF shared).
// This is the default pairwise potential used by both UFF and SPFF.
// Reference: UFF.cl:1170, 1185  getLJQH(dp, REQK, R2damp)
#define NB_PAIR_LJQH(dp, REQK, R2damp)  getLJQH(dp, REQK, R2damp)

// NB_PAIR_MORSE — Morse potential variant (future use).
// #define NB_PAIR_MORSE(dp, REQK, R2damp)  getMorseQ(dp, REQK, R2damp)

// ======================================================================
//  Axis 2 — Exclusion strategy macros
// ======================================================================
//
// Each exclusion variant must define ALL of:
//   NB_EXCL_ARGS        — extra kernel arguments (can be empty)
//   NB_EXCL_SETUP(iaa)  — load exclusion data for atom iaa into local vars
//   NB_EXCL_TEST(ja)    — returns bool: true if (ja,iG) is bonded (non-PBC)
//   NB_EXCL_PBC_TEST(ja,ipbc) — returns bool: true if this PBC image is excluded
//
// The template uses them like:
//   NB_EXCL_SETUP(iaa);
//   ...
//   if(bPBC){
//       ... if(!NB_EXCL_PBC_TEST(ja,ipbc)){ fe += NB_PAIR_FORCE(...); }
//   } else if(!NB_EXCL_TEST(ja)){ fe += NB_PAIR_FORCE(...); }

// ---- Variant: NEIGHS4 (4-neighbor int4, FireCore getNonBond style) ----
// Reference: UFF.cl:1095-1096, 1150, 1162-1167
// Each atom has up to 4 bonded neighbors stored in int4 neighs[iaa],
// with corresponding PBC cell indices in int4 neighCell[iaa].

#define NB_EXCL_ARGS_NEIGHS4  \
    __global int4*    neighs,       \
    __global int4*    neighCell

#define NB_EXCL_SETUP_NEIGHS4(iaa)  \
    const int4 ng  = neighs   [iaa];  \
    const int4 ngC = neighCell[iaa]

#define NB_EXCL_TEST_NEIGHS4(ja)  \
    ((ja==ng.x)||(ja==ng.y)||(ja==ng.z)||(ja==ng.w))

#define NB_EXCL_PBC_TEST_NEIGHS4(ja,ipbc)  \
    (  ((ja==ng.x)&&(ipbc==ngC.x)) \
     ||((ja==ng.y)&&(ipbc==ngC.y)) \
     ||((ja==ng.z)&&(ipbc==ngC.z)) \
     ||((ja==ng.w)&&(ipbc==ngC.w)) )

// ---- Variant: EXCL_LIST (packed sorted exclusion list, ex2 style) ----
// Reference: UFF.cl:1208+ getNonBond_ex2
// Uses a packed sorted exclusion list with int* excl and per-atom offsets.
// TODO: implement when needed — requires excl[] + excl_offs[] args.
// For now, stub that falls back to NEIGHS4 behavior.
// #define NB_EXCL_ARGS_EXCL_LIST  ...
// #define NB_EXCL_SETUP_EXCL_LIST(iaa)  ...
// #define NB_EXCL_TEST_EXCL_LIST(ja)  ...
// #define NB_EXCL_PBC_TEST_EXCL_LIST(ja,ipbc)  ...
