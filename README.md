# hypviz

Interactive visualizations of hyperbolic machine learning — drag points on the
Poincaré disk and watch the Lorentz hyperboloid respond in real time.

**Status: early development (M1).**

## Architecture

- **Python numpy kernel** (`src/hypviz/kernel/`) — dimension-agnostic source of truth.
  All operations are implemented once in Lorentz coordinates (the computational hub);
  Poincaré ball, Klein ball, and half-plane are coordinate charts on top.
- **TypeScript mirror kernel** (`runtime/`) drives the real-time browser loop
  (three.js, single stack: orthographic 2D + perspective 3D), validated against
  golden vectors generated from the Python kernel.
- **`Scene` API** emits self-contained interactive HTML; a matplotlib backend
  exports publication-grade SVG of any scene state.

## v1 scenes

1. Geodesics & distance
2. Poincaré ↔ Lorentz model correspondence
3. exp / log maps
4. Möbius addition
