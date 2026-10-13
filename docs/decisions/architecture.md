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
