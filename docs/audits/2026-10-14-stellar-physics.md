# Stellar physics audit: galaxy-sim

Scope: `src/system/SystemGen.js:61-110` (spectralClass, luminosity, temperature, evolveStar) and `src/render/shaders/render.wgsl:64-110` (starState). Star card shows class, mass and Teff (`src/main.js:72-77`). IMF reaches 100 M☉ (`src/physics/InitialConditions.js:18`).

Ranked by how much the error changes what a viewer sees or reads on the star card.

| # | Item | Code value | Literature value | Source | Verdict | Suggested change |
|---|------|-----------|------------------|--------|---------|------------------|
| 1 | Low-mass Teff + K/M boundary | 4100·(M/0.57)^0.25; K/M at 3700 K. 0.5 M☉ → 3968 K "K". Every star 0.38-0.57 M☉ reads class K | 0.50 M☉ = M1V 3660 K; M0V = 0.57 M☉ 3850 K; K9V 3930 K. K/M midpoint 3890 K | Mamajek table v2022.04.16 https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt | off by +8% at 0.5 M☉; class wrong for 0.38-0.57 M☉ (Kroupa peak) | K/M boundary 3890 K, and below 1 M☉ interpolate log T in log M through (0.10,2810)(0.16,3060)(0.27,3270)(0.40,3470)(0.50,3660)(0.57,3850)(0.64,4100)(0.70,4440)(0.78,4830)(0.88,5270)(1.0,5770) |
| 2 | Supernova flash brightness (shader) | 5e4 L☉ peak, while the giant just before it is 60·L_MS (8 M☉: 1.2e5, 20 M☉: 3.0e6). The flash dims the star | Type II SN peak M_V −14 to −18.5, about 3e7-2e9 L☉ | de Jaeger+2019 https://arxiv.org/abs/1909.13813 ; Anderson+2014 https://arxiv.org/abs/1403.7091 | wrong: flash is fainter than its progenitor | Peak lum ≥ 1e8 L☉, or at minimum 20× the end-of-giant lum |
| 3 | Main-sequence lifetime | 10 Gyr·M^-2.5 | t_H (non-rot, Z=0.014): 1 M☉ 8540, 1.5: 2242, 2: 1009, 3: 321, 5: 88, 9: 26.3, 20: 7.74, 40: 4.44, 60: 3.53 Myr. Code: 10000, 3630, 1770, 642, 179, 41, 5.6, 0.99, 0.36 | Ekström+2012 Tables 2-3, https://arxiv.org/abs/1110.5049 | off by +17% (1), ×1.8-2.0 too long (3-5), ×0.7 (20), ×10 too short (60) | 8540·M^-3.6 (M<1.25, from Table 2 points 0.8-1.25), then Ekström Table 3: 7798·M^-2.926 (1.25-3), 4375·M^-2.405 (3-7), 1274·M^-1.765 (7-15), 89.9·M^-0.808 (>15) Myr. Nothing below 0.8 M☉ is constrained; those lifetimes exceed the Hubble time anyway |
| 4 | White-dwarf cooling | 14000/(1+2e-4·t) from t=0. At cooling age 2.42 Gyr: 9437 K; 1.13 Gyr: 11423 K | 0.6 M☉ DA: ~100 kK at birth; 30000 K at 8.9 Myr, 14000 K at 245 Myr, 10000 at 633 Myr, 8000 at 1.13 Gyr, 6000 at 2.42 Gyr, 4000 at 9.9 Gyr | Montreal cooling tables (Bédard+2020) https://www.astro.umontreal.ca/~bergeron/CoolingModels/Tables/Table_Mass_0.6 | off by up to +57% (card Teff too hot after ~0.5 Gyr; too cool before ~200 Myr) | T = min(1e5, 66400·t_Myr^-0.297) K, fit to that table, within ±10% from 10 Myr to 10 Gyr |
| 5 | Remnant type above ~25 M☉ | All M ≥ 8 → "NS" | Single stars above ~25 M☉ (solar Z) form black holes by fallback; direct collapse above He core ~15 M☉. Uncertain by ~×2 | Heger+2003 https://arxiv.org/abs/astro-ph/0212469 | wrong label for ~23% of M≥8 remnants under the code's Kroupa slope | M ≥ 25 → class "BH", lum 0, no surface temperature |
| 6 | Giant-phase duration | 0.12·t_MS + 15 Myr. 20 M☉: MS 5.6 Myr, giant 15.7 Myr (74% of life red). 60 M☉: MS 0.36, giant 15.0 (98% red) | Post-MS / t_H: 0.11-0.15 for 9-40 M☉ (20 M☉: t_He 0.87 vs t_H 7.74 Myr), but 0.24 at 5 M☉ and 0.40 at 3 M☉ | Ekström+2012 Table 2, https://arxiv.org/abs/1110.5049 ; tracks VizieR J/A+A/537/A146 | physically ~18× too long at 20 M☉. The +15 Myr equals the shader's snDur, commented as a ~1 s real-time stretch, so it is likely a deliberate visibility floor. The 0.12 coefficient is also 2-3× low for 3-5 M☉ | Confirm intent before changing. Physical: tGiant = tMS·(0.4 at 3 M☉, 0.24 at 5, 0.12 at ≥9). Dropping the floor makes a 20 M☉ giant last ~3 frames |
| 7 | Giant end luminosity (×60) | L_end = 60·L_MS at all masses | Sun: RGB tip ~2300 L☉, AGB tip ~5200 L☉. Massive: L_max/L_ZAMS = 4 (15, 20 M☉), 2 (40 M☉), 10 (9 M☉) | Sackmann+1993 abstract https://stafforini.com/works/sackmann-1993-our-sun-iii/ ; Ekström+2012 tracks | off by ~40× low (1 M☉), ~15× high (20 M☉: 3.0e6 vs 1.5e5 L☉) | L_end = max(3·L_MS, 2300·M) L☉ (my fit to these anchors: 1 M☉ 2300, 9 M☉ 2.1e4, 20 M☉ 1.5e5) |
| 8 | Physical giant radius growth (from L=R²T⁴) | √60·(T_MS/3400)²: 1 M☉ ×22, 5 ×154, 10 ×354, 20 ×815 | Sun RGB tip ~170 R☉, AGB ~213 R☉ (×170-210). Massive R_max/R_ZAMS: 9 M☉ 161, 15 M☉ 164, 20 M☉ 176, 40 M☉ 199 (R_max 545-1557 R☉) | Sackmann+1993; Ekström+2012 tracks; Levesque+2005 RSG radii ≤1500 R☉ https://arxiv.org/abs/astro-ph/0504337 | off by ~8× small (1 M☉), ~4.6× large (20 M☉). Follows from row 7 | Fix row 7; L_end = max(3·L_MS, 2300·M) gives 1 M☉ ×139, 20 M☉ ×182 |
| 9 | High-mass Teff (>20 M☉) | 5778·M^0.6: 40 M☉ 52,800 K, 60 M☉ 67,400 K, 100 M☉ 91,600 K | O3V (59 M☉) 44,900 K; ~40 M☉ ≈ 40,900 K | Mamajek table | off by +29% at 40 M☉, +50% at 60 M☉ (card text; colour clamps at 40 kK) | T = min(5778·M^0.6, 33300·(M/20)^0.27) |
| 10 | Mass-luminosity | 0.23M^2.3 / M^4 / 1.4M^3.5. log L code vs Eker: 0.5 M☉ −1.20 vs −1.48; 5 M☉ 2.59 vs 2.87; 10 M☉ 3.65 vs 3.97 | Eker six-piece: 2.028logM−0.976 (0.179-0.45); 4.572logM−0.102 (0.45-0.72); 5.743logM−0.007 (0.72-1.05); 4.329logM+0.010 (1.05-2.4); 3.967logM+0.093 (2.4-7); 2.865logM+1.105 (7-31) | Eker+2018 Table 4, https://arxiv.org/abs/1807.02568 | vs Eker: off by +0.28 dex (×1.9) at 0.5 M☉ (+0.19 vs Mamajek); −0.28 to −0.32 dex (×0.5) at 5-10 M☉. Not on card; moves galaxy brightness, planet T_eq, HZ | Use Eker six-piece; above 31 M☉ it over-predicts (Mamajek O3V 59 M☉ log L = 5.82) |
| 11 | Spectral boundaries O/B, A/F, F/G, G/K | O ≥30000, A ≥7500, F ≥6000, G ≥5200 | These are the standard Harvard-scheme values in most references. Mamajek subtype midpoints: O9.5V 31900 / B0V 31400 → 31650; A9V 7400 / F0V 7220 → 7310; F9.5V 5990 / G0V 5930 → 5960; G9V 5380 / K0V 5270 → 5325. B/A → 10050 | Mamajek table | OK as textbook values; refinement only. With them, B0V reads O (15.6-17 M☉) and K0V reads G (0.84-0.87 M☉) | Optional calibration: O 31650, A 7310, F 5960, G 5325. K/M 3890 is the one that matters (row 1) |
| 12 | WD luminosity | 0.004 L☉ constant | 0.6 M☉ DA: 5.9e-3 L☉ at 14000 K, 1.5e-3 at 10000 K, 1.8e-4 at 6000 K, 3.5e-5 at 4000 K (from Mbol) | Montreal table above | OK at ~13 kK; ×100 too bright for old WDs | lum = 0.0059·(T/14000)^4 |
| 13 | Neutron star surface | 40000 K, 5e-4 L☉ | Inconsistency CONFIRMED by Stefan-Boltzmann: R=10 km at 40000 K gives ~5e-7 L☉, not 5e-4. Actual surface T UNCONFIRMED: ~1e6 K for the first 1e5-1e6 yr, photon cooling after | Yakovlev & Pethick 2004 abstract only, https://arxiv.org/abs/astro-ph/0402143 | low impact: colour clamps at 40 kK, near-invisible either way | Leave as visual choice, or show "~10⁵-10⁶ K" on the card |
| 14 | Supernova window | 15 Myr (shader `snDur`) | SN II-P plateau ~100 days | de Jaeger+2019, Anderson+2014 | deliberate visual stretch (commented in code) | No change |
| 15 | Giant temperature floor | 3400 K | RSG T_min in Ekström tracks: 9 M☉ 3483 K, 15 M☉ 3573, 20 M☉ 3707, 40 M☉ 3882. Levesque+2005 scale runs warmer than older ones | Ekström+2012 tracks; Levesque+2005 | OK within ~5-13% | Optional: 3600 K for M ≥ 8 |
| 16 | NS/WD mass threshold | 8 M☉ | Min core-collapse progenitor 8.5 +1/−1.5 M☉ | Smartt+2009 https://arxiv.org/abs/0809.0403 | OK | No change |
| 17 | Visual radius exponent | 0.35·M^0.8 (display only) | Mamajek R(M) log slope: 0.88 for 0.1-1 M☉, 0.68 for 1-20 M☉ | Mamajek table | OK, 0.8 sits inside the real range | No change |

## Item 2 tabulation: Teff and log L, code vs catalogue

Catalogue Teff and log L are log-log interpolations in Mamajek's table by its mass column. t_H is Ekström non-rotating.

| M (M☉) | Teff code | Teff Mamajek | ΔT | class code | class real | log L code | log L Mamajek | log L Eker | t_MS code (Myr) | t_H Ekström (Myr) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 2653 | 2795 | −5% | M | M6 | −2.94 | −3.01 | — | 3.2e6 | — |
| 0.15 | 2937 | 3023 | −3% | M | M5 | −2.53 | −2.60 | — | 1.2e6 | — |
| 0.3 | 3492 | 3323 | +5% | M | M3.5 | −1.84 | −1.95 | −2.04 | 2.0e5 | — |
| 0.5 | 3968 | 3660 | +8% | **K** | M1 | −1.20 | −1.39 | −1.48 | 5.7e4 | — |
| 0.7 | 4665 | 4440 | +5% | K | K5 | −0.62 | −0.76 | −0.81 | 2.4e4 | — |
| 1 | 5778 | 5770 | 0% | G | G2 | 0.00 | 0.01 | −0.01 | 10000 | 8540 |
| 1.5 | 7369 | 7020 | +5% | F | F1 | 0.70 | 0.79 | 0.77 | 3630 | 2242 |
| 2 | 8758 | 8942 | −2% | A | A1 | 1.20 | 1.41 | 1.31 | 1770 | 1009 |
| 3 | 11170 | 11348 | −2% | B | B8-B9 | 1.82 | 2.00 | 1.99 | 642 | 321 |
| 5 | 15176 | 16227 | −6% | B | B4 | 2.59 | 2.86 | 2.87 | 179 | 88 |
| 10 | 23003 | 24583 | −6% | B | B1.5 | 3.65 | 3.92 | 3.97 | 31.6 | ~22 (interp 9-12) |
| 20 | 34865 | 33116 | +5% | O | O9 | 4.70 | 4.81 | 4.83 | 5.6 | 7.74 |
| 40 | 52846 | ~40900 | +29% | O | O5 | 5.75 | 5.48 | — | 0.99 | 4.44 |
| 60 | 67401 | 44900 (O3V, 59 M☉) | +50% | O | O3 | 6.37 | 5.82 | — | 0.36 | 3.53 |

## Open questions

- **NS surface temperature (row 13)** stays UNCONFIRMED. I read only the Yakovlev & Pethick 2004 abstract and a search summary, not the cooling-curve figures.
- **SN peak luminosity (row 2)** uses V-band absolute magnitudes. Bolometric correction near peak is not applied, so the L☉ range is approximate.
- **Black-hole threshold (row 5)**: ~25 M☉ comes from the Heger+2003 text via search summary, which also says it is uncertain by ~×2. Sukhbold+2016 finds non-monotonic "islands" of explodability; not checked.
- **Solar post-MS duration**: Sackmann+1993 abstract gives ~0.6 Gyr on the RGB. The total subgiant-to-AGB span was not confirmed, so the 0.12·t_MS term for 1 M☉ (1.2 Gyr) is plausible but unverified.
- **Giant-phase floor (row 6)**: whether +15 Myr is intentional should be confirmed with the author; the shader comment on snDur suggests it is. Ekström 1-2 M☉ tracks stop before the RGB tip, so they were not used for low-mass L_max or R_max.
- **Row 7 formula** `max(3·L_MS, 2300·M)` is my fit to the anchors in that row, not a published relation.
- **Mamajek A-star masses** are non-monotonic (A3V 1.86, A4V 1.93), so the 1.5-2 M☉ interpolations carry ~±300 K noise.
