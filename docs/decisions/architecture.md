# Architecture Decisions

### [APPROVED] Two-scene crossfade for solar-system zoom
**Date:** 2026-07-08
**Context:** Adding star selection + zoom-in to procedurally generated solar systems (planets, moons, asteroid belt).
**Decision:** Separate AU-scale system scene crossfaded in below a camera-radius threshold; galaxy stays live behind as parallax backdrop. Impostor-sphere rendering (no mesh pipeline); system seed from existing per-star stellar-buffer fields; plausible-ish generation rules. Full spec: `docs/plans/2026-07-08-solar-system-zoom-design.md`.
**Alternatives considered:** Single continuous scene with camera-relative/log-depth rendering (still requires faking the ~10⁻⁹ scale ratio, adds precision plumbing for nothing); modal hard-switch viewer (breaks continuous-zoom UX).
---
