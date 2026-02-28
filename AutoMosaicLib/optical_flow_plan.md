# Optical Flow Interpolation for AutoMosaicLib

## Goal
The user observed flickering or omitted mosaics in video processing due to YOLO model occasionally missing frames (a false negative). To fix this, we need to implement a tracking / smoothing algorithm using **Optical Flow** across a rolling frame buffer.

If an object is successfully detected, we track its movement using optical flow for the next several frames. If the model fails to detect the object in those subsequent frames, we use the optical flow-shifted mask to cover the area. If the model *does* detect the object, we update the tracking anchor with the new detection.

## Proposed Strategy

1. **Frame Buffering**: Instead of processing and immediately writing every frame to the FFmpeg pipe, we maintain a `Queue<(byte[] rawData, Mat frame, List<SegmentResult> detections)>` with a size of `N` (e.g., 10 frames).
    - Alternatively, we process the frames on the fly, but we maintain the *previous frames in grayscale* for calculating optical flow. We only need the previous frame to calculate dense optical flow to the current frame.
2. **Dense vs Sparse Optical Flow**:
    - **Dense**: `calcOpticalFlowFarneback`. Calculates flow for every pixel. Excellent for shifting a full binary/float mask exactly where it needs to go (`remap` function). Computationally heavier but very accurate for masks.
    - **Sparse**: `calcOpticalFlowPyrLK`. Calculates flow for keypoints. We would need to extract keypoints from the bounding box of the detection, track them, and then translate the bounding box/mask based on average movement.
    - **Decision**: Since we are tracking non-rigid masks (genitals), dense flow with `remap` is usually the most accurate way to warp the previous mask to the new frame. Or, we can use bounding-box tracking. For simplicity and performance, tracking a slightly expanded bounding box or just shifting the mask center might be enough, but warping the mask via Farneback is the most robust visually. Let's use `calcOpticalFlowFarneback` on a downscaled grayscale frame to keep performance up.
    
    *Wait, Farneback on 1080p is slow. We should calculate it on a downscaled image (e.g., 480p) and then scale the flow matrix up.*

3. **Algorithm (Per Frame)**
    - Decode Frame $F_t$ (current frame).
    - Convert $F_t$ to grayscale $G_t$ and resize to a smaller resolution (e.g., width 640) for speed.
    - Run YOLO on $F_t$ to get `detections_t`.
    - If `detections_t` has objects:
        - We apply these detections.
        - We save the combined mask $M_t$ and the gray frame $G_t$ to use as the baseline for the next frame.
        - Reset a `missing_frames_counter` to 0.
    - If `detections_t` is empty (or missing previously tracked objects):
        - Check if we have $M_{t-1}$ and $G_{t-1}$ and `missing_frames_counter < MAX_BUFFER` (e.g., 10).
        - If yes:
            - Calculate Optical Flow between $G_{t-1}$ and $G_t$.
            - Use the flow map to `remap` (warp) $M_{t-1}$ to the new frame, creating our interpolated mask $M_t$.
            - Apply mosaic using this interpolated $M_t$.
            - Update $M_{t-1}$ to be this new $M_t$, and $G_{t-1}$ to $G_t$.
            - Increment `missing_frames_counter`.
        - If no (exceeded buffer or no history):
            - No mosaic applied.

4. **Refined Object-Level Tracking vs Global Mask Tracking**
    - The easiest implementation is **Global Mask Tracking**: we take the boolean union of all detected masks in frame $t-1$, and if frame $t$ has fewer detections, we warp the entire mask union using optical flow.
    - However, if there are *multiple* objects, and YOLO detects one but loses the other, global tracking is tricky to mix with new detections.
    - **Mix Strategy**: 
      - Calculate the warped previous mask: `Warped_M = remap(M_{t-1}, Flow)`.
      - Calculate the new mask from YOLO: `New_M`.
      - Final Mask = `Warped_M OR New_M`.
      - BUT, we don't want the warped mask to persist forever if the object actually left the screen. We need to gradually decay the warped mask, or hard-stop it after `N` frames.
      - We can store an age for each "blob" in the mask, but that requires connected components.
      - A simpler mechanism: maintain a list of `TrackedMask` objects. Each has a `Mat Mask` and an `int Age`.

### Alternative: Tracking API (KCF, CSRT)
OpenCV has a Tracking API (`cv2.TrackerCSRT_create`). C# OpenCvSharp has `TrackerCSRT`.
Instead of dense optical flow on the whole frame, we can initialize a Tracker for each detected bounding box.
If YOLO misses the box, the Tracker provides the new bounding box. We can then just apply the mosaic to the tracked bounding box (or shift the previous mask to the new bounding box center).

Given the user specifically requested "オプティカルフローのような技術で動きを推論" (infer movement using technologies like optical flow), Farneback Dense Optical Flow is a direct answer to their request and very robust for organic movements.

### Proposed Implementation Details (Dense Flow)
- Variables in `VideoProcessor`:
  - `Mat prevGray` (downscaled)
  - `Mat prevMask` (full resolution, float32, 0.0 to 1.0)
  - `int framesSinceLastDetection`
- Pipeline per frame $F$:
  1. $G$ = resize $F$ to width 640. Convert to Grayscale.
  2. YOLO predict $F \rightarrow$ `results`.
  3. Draw `results` masks into a new `Mat currentMask` (float32, 1.0 for mask, 0.0 for background).
  4. If `prevGray` exists:
     a. Calculate Flow `calcOpticalFlowFarneback(prevGray, G)`.
     b. Resize Flow matrix to match $F$'s full resolution, multiply flow vectors by `scaleX, scaleY`.
     c. `cv2.remap(prevMask)` using the scaled flow to get `warpedPrevMask`.
  5. Combine masks:
     a. If `results.Count > 0`: `framesSinceLastDetection = 0; finalMask = currentMask OR warpedPrevMask;` (Wait, if YOLO detects it, maybe we don't need warped mask? Or we union them. Let's union them to fill gaps, but only if `framesSinceLastDetection < BufferSize`).
     b. Actually, we only want to use the warped mask for *missing* parts. A simple `cv2.max(currentMask, warpedPrevMask)` works.
     c. To decay, we can multiply `warpedPrevMask` by $0.9$ each frame. If it drops below 0.5, it becomes 0.
     d. Or simply track a counter. If `results.Count == 0`, `framesSinceLastDetection++`. If `results.Count > 0`, `framesSinceLastDetection = 0`. If `framesSinceLastDetection > 10`, `warpedPrevMask = 0`.
  6. Apply mosaic using `finalMask`.
  7. `prevGray = G`, `prevMask = finalMask`.
