# JAX-AMR: A JAX-based adaptive mesh refinement framework

JAX-AMR is an adaptive mesh refinement framework based on dynamically updated multi-layer blocks with fixed positions and fixed shapes. This framework is fully compatible with JIT and vectorized operations.

Authors:
- [Haocheng Wen](https://github.com/thuwen)
- [Faxuan Luo](https://github.com/luofx23)

Correspondence via [mail](mailto:haochengwenson@126.com) (Haocheng Wen).

## Implementation Strategy
The multi-layer blocks and the partitioning and refinement strategies in JAX-AMR are illustrated as follows.

<img src="/docs/images/blocks in JAX-AMR.png" alt="Schematic diagram of multi-layer blocks in JAX-AMR" height="600"/>

For the detailed implementation strategies of JAX-AMR, please refer to our [paper](xxx).

## State of the Project

- [x] 2D AMR, fully jit-compiled ✅
- [x] conjuction with the CFD solver ✅
- [ ] 3D AMR (soon)
- [ ] Immerse boudary method (soon)
- [ ] parallel mannagment (soon)


## License
This project is licensed under the MIT License - see 
the [LICENSE](LICENSE) file or for details https://en.wikipedia.org/wiki/MIT_License.
