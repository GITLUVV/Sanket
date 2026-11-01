# Landmark Extraction

Sanket uses MediaPipe to extract 3D skeletal landmarks from video frames without retaining raw video footage.

## Extracted Landmarks
- **Hands (MediaPipe Hands)**:
  - 21 3D landmarks per hand $(x, y, z)$ in camera space.
  - Dual-hand tracking (Right and Left hands).
  - Confidence scoring and temporal linear interpolation for dropped frames.
- **Upper Body (MediaPipe Pose)**:
  - 33 3D pose landmarks.
  - Key reference anchors extracted: Nose ($L_0$), Left/Right Shoulders ($L_{11}, L_{12}$), Left/Right Elbows ($L_{13}, L_{14}$), Left/Right Wrists ($L_{15}, L_{16}$), and computed Sternum center.

## Coordinate Normalization
To eliminate camera distance and signer position bias:
1. **Wrist Origin**: Hand landmarks are translated so the wrist ($L_0$) is at $(0, 0, 0)$.
2. **Palm Metacarpal Scale**: Coordinates are scaled by the Euclidean distance between wrist ($L_0$) and middle MCP knuckle ($L_9$), ensuring $\|L_9 - L_0\|_2 = 1.0$.
3. **Invertibility**: Saved metadata allows exact mathematical reconstruction.
