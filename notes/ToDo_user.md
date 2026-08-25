

* General desing - gui should be optimize for debugging different forcefields - it should efficiently bind and visualize arrays of atoms and bonds while these can come form diffrent forcefield. We need to establish uniform representation shared between different forcefield so they can communicate even thogu hinternal representation of UFF, RigidAtomFF and RigidMole FF is very different


* Main focus at the start should be on RigidAtomFF RAFF
* Key is to make efficient memory layour which is optimized both for bonin-and nonbonding interactions, where
    * Bonding are represented by atom-frame ports which are rotated by quaternion rigid body dynamics  (as-rigid-as-possible, ARAP paper) these interact with positon of fixed atom.
    * capping atoms (e.g. hydrogen, epairs) have no ports
        * thous they have independne DOF, or should they be just apendix rigidly fixed to given port of host atom?
    * There are two version: 
        1. fixed topology where each port interact with just one neighbor atom (i.e. 1-to-1 bijective map port k of atom i connects to atom j) via harmonic potential
        2. reactive forcefield, where each port can interact to all atoms in proximity via dissociative potential (like Morse or its fast polynominal apporximation) 
    * Nonbonding interactions are accelerated by AABB bounding boxes for fragments optimized to GPU workrougpise or local memory (16,32,64,128 atoms per fragment).
       * Design decision should be made if we organize fragment continous in memory - can be faster, but we also may want to organize node atoms and coping atom
       * For fast collision we use spacial linearized (harmonic spring-like E=k(|ri-rj|-R0ij)^2 for short distace, which transition to polynominal Morse approx at far distance)
* Projective Dynamics and position based dynamics - optimized for relaxation ratherthan molecular dynamics 


* import OpenCL kernel to kiksctart from SPAMMM and FireCore
* make codemaps of other repos to know where to import what