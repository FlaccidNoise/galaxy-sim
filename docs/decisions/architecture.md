# Architecture Decisions

### [APPROVED] Two-scene crossfade for solar-system zoom
**Date:** 2026-07-08
**Context:** Adding star selection + zoom-in to procedurally generated solar systems (planets, moons, asteroid belt).
**Decision:** Separate AU-scale system scene crossfaded in below a camera-radius threshold; galaxy stays live behind as parallax backdrop. Impostor-sphere rendering (no mesh pipeline); system seed from existing per-star stellar-buffer fields; plausible-ish generation rules. Full spec: `docs/plans/2026-07-08-solar-system-zoom-design.md`.
**Alternatives considered:** Single continuous scene with camera-relative/log-depth rendering (still requires faking the ~10⁻⁹ scale ratio, adds precision plumbing for nothing); modal hard-switch viewer (breaks continuous-zoom UX).
---

### [APPROVED] Star-relative render frame while a star is selected
**Date:** 2026-10-13
**Context:** With a star selected, the camera target was copied from the CPU position snapshot, which refreshes once per tree-rebuild round trip (many frames). The star renders from the live GPU buffer every frame, so the target lagged and the view wobbled; at radius ≤ 1.5 the star drifted off-centre by most of the viewport.
**Decision:** The galaxy render shaders subtract `particles[selectedIdx].pos` from every particle position, so the selected star is the origin of the render frame and the camera target is `[0,0,0]` while selected. No per-frame CPU readback is involved; tracking is exact every frame. Picking subtracts the same offset from the snapshot. On select/deselect the camera target is converted between frames so the view does not snap.
**Alternatives considered:** Per-frame position readback (adds a GPU→CPU stall per frame); velocity extrapolation of the snapshot (needs a velocity readback and still drifts between snapshots); slowing the sim while selected (hides the symptom only).

---

### [APPROVED] Planets live in physical AU, rendered under one per-system scale
**Date:** 2026-10-14
**Context:** The system scene inflates the star (0.35 units for the Sun, ~19× its true radius in AU) so it reads on screen. The generator added three visual star radii to the first orbit and then compared that rendered distance against the physical frost line, so around stars under half a solar mass every planet counted as past the frost line and became a giant. Sun-like stars could never form scorched or Venusian worlds.
**Decision:** The orbit ladder, frost line, habitable zone, equilibrium temperature and kinds are computed in physical AU (`planet.au`). One uniform factor per system (`max(1, 3·R_ms / au₀)`) maps the ladder to rendered units (`planet.a`), so relative distances are preserved and the innermost orbit clears the inflated star. Orbit rates use rendered distance, which is where the watchability tuning lives. The shader takes the rendered frost line as a uniform instead of reconstructing it from star temperature. Stellar laws (`luminosity`, `temperature`, lifecycle) are shared between SystemGen and render.wgsl; age reaches the generator as an argument and never enters the seed. Planet positions come from a Kepler solve with three fixed Newton steps, identical on CPU and GPU, so clicks land on the drawn planet.
**Alternatives considered:** Shrinking the star to true scale (planets become invisible dots at any zoom that shows the star); an additive star-radius offset in rendered units with physical classification (compresses inner ratios); solving Kepler to convergence on each side (CPU and GPU would disagree in the last iteration).
