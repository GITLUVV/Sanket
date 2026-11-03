# SANKET (संकेत) — Accessible UI & Visual Design System

> **Visual Interface, High-Contrast Palette, Skeletal Overlays, and Subtitle Guidelines**  
> *Engineered to maximize readability and accessibility for Deaf and Hard of Hearing (DHH) users.*

---

## 1. Design Philosophy & Accessibility Standards

The visual interface of SANKET is built on the principles of **Universal Design** and the **Web Content Accessibility Guidelines (WCAG 2.1 AAA)**:

1. **Immediate Feedback**: Users must see instant visual confirmation of hand tracking status, landmark stability, and classification confidence within the same visual field.
2. **High-Contrast Separation**: Overlays must remain clearly legible against diverse visual backgrounds (light walls, skin tones, patterned clothing, dark environments).
3. **Dual-Language Accessibility**: Sign language glosses are displayed simultaneously in English capital glosses and Hindi Devanagari script (`docs/research/INITIAL_VOCABULARY.md`).
4. **Zero Flashing / Flicker**: Visual state changes must not oscillate or flash rapidly (adheres to photosensitive seizure prevention standards).

---

## 2. High-Contrast Color Palette

All UI overlay colors in OpenCV (BGR format) are chosen for high visibility and color-blind friendliness:

| Semantic Element | Hex Color | BGR Tuple | Usage in Overlay |
| :--- | :--- | :--- | :--- |
| **Dominant Hand Skeleton** | `#00E5FF` (Cyan) | `(255, 229, 0)` | Bones and joints for the primary/dominant signing hand. |
| **Non-Dominant Hand** | `#76FF03` (Lime Green) | `(3, 255, 118)` | Bones and joints for the base/secondary signing hand. |
| **Active Gloss Banner** | `#1A237E` (Deep Navy) | `(126, 35, 26)` | Solid background banner for predicted sign text. |
| **High Confidence** ($\ge 0.75$) | `#00E676` (Bright Green) | `(118, 230, 0)` | Confidence bar fill when classification is firm. |
| **Medium Confidence** ($0.50\text{--}0.74$) | `#FFD600` (Amber Yellow)| `(0, 214, 255)` | Warning state for ambiguous gesture alignment. |
| **Low / OOV Gating** ($< 0.50$) | `#FF1744` (Coral Red) | `(68, 23, 255)` | Indicates `NO_SIGN`, `IDLE`, or out-of-vocabulary rejection. |
| **HUD Telemetry Text** | `#FFFFFF` (Pure White) | `(255, 255, 255)` | Operational statistics (FPS, latency ms, tracking state). |
| **Text Shadow / Stroke** | `#000000` (Pure Black) | `(0, 0, 0)` | 2px outline surrounding all rendered text for contrast. |

---

## 3. Video Frame Layout & Overlay Geometry

For the standard $640 \times 480$ camera frame:

```text
+-------------------------------------------------------------------------+
| [TOP BANNER]                                          [TELEMETRY HUD]   |
| GLOSS: HELLO (नमस्ते)                                   FPS: 31.2         |
| CONF: [████████████████████░░░░] 84%                  LATENCY: 18.4 ms  |
|                                                       TRACKING: 2 HANDS |
+-------------------------------------------------------------------------+
|                                                                         |
|                     (Skeletal Hand Overlay)                             |
|                           * * *                                         |
|                          *  |  *                                        |
|                            ---                                          |
|                                                                         |
+-------------------------------------------------------------------------+
| [CANDIDATE BARS]                                                        |
| 1. HELLO        84% [████████████████░░░░]                              |
| 2. HELP          9% [██░░░░░░░░░░░░░░░░░░]                              |
| 3. THANK_YOU     4% [█░░░░░░░░░░░░░░░░░░░]                              |
+-------------------------------------------------------------------------+
| [ACCESSIBLE SUBTITLE TRANSCRIPT]                                        |
| Transcript: HELLO, PLEASE HELP WATER                                    |
+-------------------------------------------------------------------------+
```

### Overlay Regions
1. **Top Banner (y = 10 to 60px)**:
   - Primary predicted gloss in bold font.
   - Confidence percentage meter.
2. **Telemetry HUD (Top Right, x = 450 to 630px)**:
   - Camera FPS, inference FPS, and end-to-end latency in milliseconds.
3. **Candidate Bars (Bottom Left, y = 350 to 420px)**:
   - Top-3 candidate predictions showing diagnostic probability distribution.
4. **Accessible Subtitles (Bottom Banner, y = 430 to 475px)**:
   - Rolling transcript history buffer representing recognized communication sequence.

---

## 4. Typography & Font Specifications

- **Font Face**: OpenCV `FONT_HERSHEY_DUPLEX` for headers and labels; `FONT_HERSHEY_SIMPLEX` for small telemetry numbers.
- **Stroke & Scaling**:
  - Main Gloss Banner: Scale `0.9`, Thickness `2`, with a `3px` black backing box.
  - Telemetry & Subtitles: Scale `0.5`, Thickness `1`.
- **Text Dropshadows**: All text strings are drawn twice: first in pure black with a 1px offset to produce an outline, then in foreground color.

