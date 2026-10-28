# Initial ISL Constrained Vocabulary Specification — Sanket

- **Issue Reference**: SANKET-010 (Initial ISL vocabulary definition)
- **Status**: Formally Specified Baseline
- **Date**: 2026-10-28
- **Linguistic Framework**: Indian Sign Language (ISL) Phonological Parameters

---

## 1. Linguistic Foundation & Selection Principles

Indian Sign Language (ISL) is a natural visual-spatial language with its own independent grammar, morphology, and syntax, completely distinct from spoken Hindi, spoken English, American Sign Language (ASL), or British Sign Language (BSL).

In sign language phonology (Stokoe / Battison model adapted for ISL), every sign is composed of five distinct articulatory parameters:
1. **Handshape (*Tab*)**: The geometric configuration of fingers and thumb.
2. **Location (*Dez*)**: The spatial location on the body or in neutral signing space where the sign is articulated.
3. **Movement (*Sig*)**: The path, trajectory, speed, and repetition of the hands.
4. **Palm Orientation**: The spatial direction in which the palm and metacarpals face.
5. **Non-Manual Markers (NMM)**: Facial expressions, head tilts, and mouth shapes that convey grammatical mood or negation.

### Selection Criteria for Phase 1:
- **High Utility**: Prioritize everyday communication, emergency requests, courtesy phrases, and basic needs.
- **Morphological Diversity**: Balanced representation across one-handed and two-handed signs with varied spatial locations.
- **Landmark Discriminability**: Select signs with pronounced geometric distinctiveness to ensure robust baseline separation on classical classifiers (Random Forest, SVM) before temporal models are introduced.
- **INCLUDE Dataset Alignment**: Ensure all selected classes exist in the primary benchmark dataset (**INCLUDE**) to enable zero-overhead offline training in Phase 1.

---

## 2. Master Constrained Vocabulary (16 Signs)

| ID | Class Label / Gloss | Semantic Tier | Handedness | Dominant Handshape | Articulation Zone | Movement Type | INCLUDE Match |
| :- | :------------------ | :------------ | :--------- | :----------------- | :---------------- | :------------ | :------------ |
| **01** | `HELLO` / `NAMASTE` | Greetings | Two-Handed | Open Flat Palm (B-hand) | Chest / Midline | Static Contact (Anjali) | Yes (`Hello`) |
| **02** | `THANK_YOU` | Courtesy | One-Handed | Open Flat Palm (B-hand) | Chin → Forward | Linear Forward Stroke | Yes (`Thank You`) |
| **03** | `PLEASE` | Courtesy | One-Handed | Open Flat Palm (5-hand) | Chest | Circular Rubbing Motion | Yes (`Please`) |
| **04** | `YES` | Affirmation | One-Handed | Closed Fist (S-hand) | Neutral Chest Space | Vertical Nodding Tilt | Yes (`Yes`) |
| **05** | `NO` | Negation | One-Handed | Extended Index (1-hand) | Neutral Space | Lateral Oscillating Shake | Yes (`No`) |
| **06** | `HELP` | Emergency | Two-Handed | Dominant Fist on Palm | Neutral Chest Space | Upward Elevation Stroke | Yes (`Help`) |
| **07** | `WATER` | Daily Need | One-Handed | Cupped Hand / W-shape | Mouth / Chin | Inward Tap to Lips | Yes (`Water`) |
| **08** | `FOOD` / `EAT` | Daily Need | One-Handed | Flattened O-hand (Pinch) | Mouth | Repeated Contact to Lips | Yes (`Food`) |
| **09** | `DOCTOR` | Healthcare | Two-Handed | 2-Finger Fork on Wrist | Non-Dominant Wrist | Downward Pulse Tap | Yes (`Doctor`) |
| **10** | `TIME` | Query | One-Handed | Extended Index (1-hand) | Non-Dominant Wrist | Tap on Wrist Back | Yes (`Time`) |
| **11** | `WHERE` | Query | Two-Handed | Open Supine Palms (5-hand)| Lower Chest Space | Lateral Shrug Oscillation | Yes (`Where`) |
| **12** | `STOP` | Directive | Two-Handed | Dominant Knife-Hand | Mid Chest Space | Sharp Striking Contact | Yes (`Stop`) |
| **13** | `NAME` | Identity | Two-Handed | H-hand (Index + Middle) | Upper Neutral Space | Crosswise Tapping Contact | Yes (`Name`) |
| **14** | `GOOD` | Evaluation | One-Handed | Thumbs-Up (A-hand) | Neutral Chest Space | Static / Upward Thrust | Yes (`Good`) |
| **15** | `BAD` | Evaluation | One-Handed | Thumbs-Down / Palm Flip | Lower Neutral Space | Downward Pronation Flip | Yes (`Bad`) |
| **16** | `ME` / `I` | Pronoun | One-Handed | Extended Index (1-hand) | Center of Chest | Inward Pointing Contact | Yes (`Me`) |

---

## 3. Detailed Articulatory & Landmark Profiles

### Sign 01: `HELLO` (`NAMASTE`)
- **Handedness**: Two-Handed Symmetrical.
- **Handshape**: Both hands in open flat configuration (all five fingers extended, adducted).
- **Location**: Sternum level, midline of torso, approximately 10–15 cm from chest.
- **Orientation**: Palms touching facing each other; fingertips pointing vertically upward ($+Y$).
- **Movement**: Static hold or brief gentle inward compression.
- **Feature Pipeline**: Dual-hand tracking ($21 \times 2 = 42$ landmarks). Distance between left and right palm centers $< 0.05$ normalized units.

### Sign 02: `THANK_YOU`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Open flat hand, fingers together, thumb alongside index finger.
- **Location**: Initial contact at center of chin / lower lip, moving forward into neutral signing space.
- **Orientation**: Palm facing signer initially, tilting slightly upward during extension.
- **Movement**: Linear forward stroke away from face.
- **Feature Pipeline**: Dominant hand 63 landmarks + MediaPipe Pose nose landmark anchor to verify chin proximity.

### Sign 03: `PLEASE`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Open palm with fingers slightly spread (5-handshape).
- **Location**: Center of chest (pectoral area).
- **Orientation**: Palm flat against chest facing inward.
- **Movement**: Clockwise circular rubbing trajectory on sternum.
- **Feature Pipeline**: Trajectory of wrist landmark $L_0$ forms circular planar locus.

### Sign 04: `YES`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Closed fist, thumb folded across fingers (S-handshape).
- **Location**: Neutral signing space in front of dominant shoulder.
- **Orientation**: Palm facing forward or slightly inward.
- **Movement**: Rhythmic vertical flexion of wrist (up and down tilting like a nodding head).
- **Feature Pipeline**: Oscillating angle between forearm vector and metacarpal vector.

### Sign 05: `NO`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Extended index finger or extended index and middle fingers, thumb closed.
- **Location**: Chest height in neutral space.
- **Orientation**: Palm facing outward or forward.
- **Movement**: Rapid horizontal lateral oscillation (side-to-side wagging).
- **Feature Pipeline**: Periodic change in horizontal $X$-coordinate of index tip ($L_8$).

### Sign 06: `HELP`
- **Handedness**: Two-Handed Asymmetric (Dominant active, Non-Dominant support).
- **Handshape**: Non-dominant hand flat supine palm; dominant hand closed fist with thumb up.
- **Location**: Chest level.
- **Orientation**: Non-dominant palm facing upward; dominant fist placed on open palm.
- **Movement**: Both hands move upward together in a lifting stroke.
- **Feature Pipeline**: Relative vertical distance between dominant wrist and non-dominant palm remains zero during positive vertical translation ($+Y$).

### Sign 07: `WATER`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: 'W' handshape (index, middle, ring extended and spread) or slightly cupped C-shape.
- **Location**: Contact at corner of mouth.
- **Orientation**: Palm facing inward or toward mouth.
- **Movement**: Repeated gentle tapping against mouth/chin.
- **Feature Pipeline**: Distance from index tip ($L_8$) to mouth corner $< 0.08$ normalized units.

### Sign 08: `FOOD` / `EAT`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Flattened O-hand (fingertips gathered touching the thumb tip).
- **Location**: Lips / oral cavity.
- **Orientation**: Palm facing toward face.
- **Movement**: Rhythmic double tapping of grouped fingertips toward mouth.
- **Feature Pipeline**: Low Euclidean distance between thumb tip ($L_4$) and finger tips ($L_8, L_{12}, L_{16}, L_{20}$) combined with proximity to mouth landmarks.

### Sign 09: `DOCTOR`
- **Handedness**: Two-Handed Asymmetric.
- **Handshape**: Non-dominant hand extended relaxed; dominant hand index and middle fingers extended (feeling pulse).
- **Location**: Volar aspect of non-dominant wrist.
- **Movement**: Dominant fingertips tap inner wrist of non-dominant arm twice.
- **Feature Pipeline**: Dominant fingertips ($L_8, L_{12}$) converge with high spatial precision on non-dominant wrist ($L_0$).

### Sign 10: `TIME`
- **Handedness**: One-Handed (Dominant) targeting non-dominant wrist.
- **Handshape**: Extended dominant index finger.
- **Location**: Dorsal aspect of non-dominant wrist (wristwatch position).
- **Movement**: Tapping motion on wrist back.
- **Feature Pipeline**: Contact between dominant index tip and non-dominant wrist. Distinct from `DOCTOR` by dorsal vs volar wrist approach.

### Sign 11: `WHERE`
- **Handedness**: Two-Handed Symmetrical.
- **Handshape**: Open flat hands with palms facing upward (supine).
- **Location**: Lower chest signing space, hands shoulder-width apart.
- **Movement**: Gentle horizontal swaying / shrugging oscillation outward.
- **Feature Pipeline**: Parallel upward palm normal vectors ($+Z$) combined with lateral separation.

### Sign 12: `STOP`
- **Handedness**: Two-Handed Asymmetric.
- **Handshape**: Non-dominant open horizontal palm; dominant knife-hand (B-shape).
- **Location**: Mid-chest level.
- **Movement**: Dominant hand strikes down sharply onto non-dominant palm.
- **Feature Pipeline**: Rapid negative $Y$-velocity of dominant hand terminating in sudden deceleration at contact point.

### Sign 13: `NAME`
- **Handedness**: Two-Handed Symmetrical.
- **Handshape**: Both hands with index and middle fingers extended (H-handshape).
- **Location**: Upper chest neutral space.
- **Movement**: Fingers cross over and tap against each other twice.
- **Feature Pipeline**: Dual H-handshape geometry and overlapping coordinate intersection of index/middle distal segments.

### Sign 14: `GOOD`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Thumbs-up posture (closed fist with thumb pointing upward).
- **Location**: Center of chest in neutral space.
- **Movement**: Static hold or brief firm forward thrust.
- **Feature Pipeline**: High vertical separation between thumb tip ($L_4$) and folded metacarpals.

### Sign 15: `BAD`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Thumbs-down posture or outward flick of palm downward.
- **Location**: Lower neutral space.
- **Movement**: Wrist pronation turning thumb downward.
- **Feature Pipeline**: Inversion of thumb tip $Y$-vector relative to wrist.

### Sign 16: `ME` / `I`
- **Handedness**: One-Handed (Dominant).
- **Handshape**: Extended index finger, other fingers folded into fist.
- **Location**: Center of signer's chest.
- **Movement**: Index finger points inward and taps chest.
- **Feature Pipeline**: Metacarpal vector points inward (negative $Z$) towards sternum pose keypoint.

---

## 4. Pairwise Confusion Risk Analysis & Disambiguation

| Potential Confusion Pair | Shared Morphology | Primary Disambiguating Geometric Features |
| :----------------------- | :---------------- | :---------------------------------------- |
| `DOCTOR` vs `TIME` | Both tap the non-dominant wrist | **Palmar vs Dorsal**: `DOCTOR` approaches volar (inner) wrist; `TIME` taps outer wristwatch surface. |
| `WATER` vs `FOOD` | Both articulated near the mouth | **Handshape**: `WATER` uses spread W-hand or cupped palm; `FOOD` gathers all 5 fingertips into a pinched O-cluster. |
| `HELLO` vs `PLEASE` | Both use flat open palms near chest | **One vs Two Hands & Motion**: `HELLO` uses both palms touching stationary; `PLEASE` uses one hand rubbing in circular trajectory. |
| `GOOD` vs `YES` | Both involve closed fist postures | **Thumb Position & Trajectory**: `GOOD` extends thumb upward statically; `YES` keeps thumb tucked and nods the entire fist vertically. |
| `STOP` vs `HELP` | Both involve two hands in contact | **Stroke Dynamics**: `STOP` is a downward striking slap; `HELP` places the fist on palm and lifts upward together. |

---

## 5. Vocabulary Freeze Protocol & Project Owner Sign-Off

This 16-sign vocabulary represents the **frozen target vocabulary for Phase 1**:
- Any addition or deletion of classes requires an updated revision of this document.
- Ingestion scripts in SANKET-020 will filter candidate video datasets strictly for these 16 class glosses.
- Signer-independent splitting in SANKET-022 will enforce complete class balance across these 16 categories.
