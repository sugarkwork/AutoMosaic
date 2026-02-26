using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace AutoMosaicLib
{
    /// <summary>
    /// Represents a single segmentation detection result.
    /// </summary>
    public class SegmentResult
    {
        /// <summary>Bounding box in original image coordinates.</summary>
        public Rect BoundingBox { get; set; }

        /// <summary>Detected class ID.</summary>
        public int ClassId { get; set; }

        /// <summary>Class name (e.g. "pussy", "penis").</summary>
        public string ClassName { get; set; } = string.Empty;

        /// <summary>Detection confidence score (0-1).</summary>
        public float Confidence { get; set; }

        /// <summary>
        /// Binary/float mask at original image resolution (single-channel, CV_8UC1).
        /// Pixel value 255 = detected region, 0 = background.
        /// </summary>
        public Mat Mask { get; set; } = new Mat();
    }

    /// <summary>
    /// YOLOv8-seg ONNX segmentation inference engine.
    /// Loads an sd.onnx model exported with NMS and performs instance segmentation.
    /// 
    /// Input:  "images"  [1, 3, 960, 960]  float32
    /// Output: "output0" [1, 300, 38]       float32  (NMS detections: x,y,w,h,conf,cls, 32 mask coeffs)
    /// Output: "output1" [1, 32, 240, 240]  float32  (mask prototypes)
    /// </summary>
    public class YoloSegmentator : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string[] _classNames;
        private bool _disposed;

        /// <summary>
        /// Set to a directory path to enable debug image output at each processing stage.
        /// Set to null to disable debug output.
        /// </summary>
        public string? DebugOutputDir { get; set; }

        // Model constants
        private const int InputSize = 960;
        private const int MaskProtoH = 240;
        private const int MaskProtoW = 240;
        private const int NumMaskCoeffs = 32;

        /// <summary>
        /// Initializes the segmentator by loading an ONNX model.
        /// </summary>
        /// <param name="modelPath">Path to the sd.onnx file.</param>
        /// <param name="classNames">Optional class names array. Defaults to ["pussy", "penis"] for sd model.</param>
        /// <param name="useGpu">If true, attempt to use CUDA execution provider.</param>
        public YoloSegmentator(string modelPath, string[]? classNames = null, bool useGpu = false)
        {
            var sessionOptions = new SessionOptions();
            if (useGpu)
            {
                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA(0);
                }
                catch
                {
                    // Fall back to CPU if CUDA is not available
                }
            }
            sessionOptions.AppendExecutionProvider_CPU(0);

            _session = new InferenceSession(modelPath, sessionOptions);
            _classNames = classNames ?? new[] { "pussy", "anus", "penis", "nipple" };
        }

        /// <summary>
        /// Gets the class names used by the model.
        /// </summary>
        public IReadOnlyList<string> ClassNames => _classNames;

        /// <summary>
        /// Performs segmentation inference on an input image.
        /// </summary>
        /// <param name="image">Input image as OpenCvSharp Mat (BGR).</param>
        /// <param name="confThreshold">Minimum confidence threshold for detections.</param>
        /// <param name="marginBlockSize">
        /// Controls mask dilation margin. The margin in pixels is calculated as
        /// max(originalW, originalH) / marginBlockSize. Set to 0 to disable dilation.
        /// </param>
        /// <returns>List of SegmentResult objects for each detection above the threshold.</returns>
        public List<SegmentResult> Predict(Mat image, float confThreshold = 0.5f, int marginBlockSize = 100)
        {
            if (image.Empty())
                throw new ArgumentException("Input image is empty.", nameof(image));

            int origH = image.Rows;
            int origW = image.Cols;

            // Prepare debug output directory
            if (DebugOutputDir != null)
            {
                Directory.CreateDirectory(DebugOutputDir);
                Console.WriteLine($"[DEBUG] Input image size: {origW}x{origH}, channels: {image.Channels()}");
            }

            // --- Preprocessing ---
            var inputTensor = Preprocess(image);

            // --- Inference ---
            var inputName = _session.InputNames[0]; // "images"
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            Console.WriteLine($"[DEBUG] Running ONNX inference...");
            using var rawResults = _session.Run(inputs);
            var resultsList = rawResults.ToList();

            // output0: [1, 300, 38] - detections
            var detectionsRaw = resultsList[0].AsTensor<float>();
            // output1: [1, 32, 240, 240] - mask prototypes
            var protoRaw = resultsList[1].AsTensor<float>();

            Console.WriteLine($"[DEBUG] output0 shape: [{string.Join(", ", detectionsRaw.Dimensions.ToArray())}]");
            Console.WriteLine($"[DEBUG] output1 shape: [{string.Join(", ", protoRaw.Dimensions.ToArray())}]");

            // Dump first few detections raw values for inspection
            if (DebugOutputDir != null)
            {
                int dumpCount = Math.Min(5, detectionsRaw.Dimensions[1]);
                for (int d = 0; d < dumpCount; d++)
                {
                    var vals = new float[detectionsRaw.Dimensions[2]];
                    for (int v = 0; v < vals.Length; v++)
                        vals[v] = detectionsRaw[0, d, v];
                    Console.WriteLine($"[DEBUG] detection[{d}]: [{string.Join(", ", vals.Select(v => v.ToString("F4")))}]");
                }
            }

            // --- Postprocessing ---
            return Postprocess(detectionsRaw, protoRaw, origW, origH, confThreshold, marginBlockSize);
        }

        /// <summary>
        /// Preprocesses the image: resize to 960x960, normalize to [0,1], and create NCHW tensor.
        /// </summary>
        private DenseTensor<float> Preprocess(Mat image)
        {
            // Convert BGR to RGB
            using var rgb = new Mat();
            Cv2.CvtColor(image, rgb, ColorConversionCodes.BGR2RGB);

            // Resize to model input size
            using var resized = new Mat();
            Cv2.Resize(rgb, resized, new Size(InputSize, InputSize));

            // Debug: save preprocessed image
            if (DebugOutputDir != null)
            {
                using var debugPreproc = new Mat();
                Cv2.CvtColor(resized, debugPreproc, ColorConversionCodes.RGB2BGR);
                Cv2.ImWrite(Path.Combine(DebugOutputDir, "00_preprocessed_960x960.png"), debugPreproc);
                Console.WriteLine($"[DEBUG] Saved preprocessed image (960x960)");
            }

            // Create tensor [1, 3, H, W]
            var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });

            // Normalize pixel values to [0, 1] and arrange in NCHW format
            unsafe
            {
                byte* ptr = (byte*)resized.Data;
                int step = (int)resized.Step();
                for (int y = 0; y < InputSize; y++)
                {
                    byte* row = ptr + y * step;
                    for (int x = 0; x < InputSize; x++)
                    {
                        int offset = x * 3;
                        tensor[0, 0, y, x] = row[offset + 0] / 255f; // R
                        tensor[0, 1, y, x] = row[offset + 1] / 255f; // G
                        tensor[0, 2, y, x] = row[offset + 2] / 255f; // B
                    }
                }
            }

            return tensor;
        }

        /// <summary>
        /// Postprocesses ONNX outputs to produce segmentation results.
        /// Matches ultralytics process_mask logic exactly.
        /// 
        /// End2end model output0 layout per detection (38 values):
        ///   [x1, y1, x2, y2, confidence, class_id, 32 mask_coefficients]
        ///   Bounding boxes are in XYXY format (not center-format) for end2end models.
        /// 
        /// Mask processing (from ultralytics ops.process_mask):
        ///   1. masks = mask_coefficients @ protos.reshape(32, -1) → reshape to (mh, mw)
        ///   2. crop_mask(masks, bboxes * ratio)  — crop raw values to bbox in mask space
        ///   3. F.interpolate bilinear to model input size
        ///   4. masks.gt_(0.0)  — threshold raw values at 0 (equivalent to sigmoid > 0.5)
        /// </summary>
        private List<SegmentResult> Postprocess(
            Tensor<float> detections,
            Tensor<float> protos,
            int origW, int origH,
            float confThreshold,
            int marginBlockSize)
        {
            var results = new List<SegmentResult>();
            int numDetections = detections.Dimensions[1]; // 300

            // Scale factors from model input (960x960) to original image
            float scaleX = (float)origW / InputSize;
            float scaleY = (float)origH / InputSize;

            // Ratio from model input to mask proto space (960 -> 240)
            float maskRatio = (float)MaskProtoW / InputSize; // 0.25

            // Count how many pass the confidence threshold
            int aboveThreshold = 0;
            for (int i = 0; i < numDetections; i++)
            {
                if (detections[0, i, 4] >= confThreshold)
                    aboveThreshold++;
            }
            Console.WriteLine($"[DEBUG] Total detections: {numDetections}, above conf {confThreshold}: {aboveThreshold}");

            int detIdx = 0;
            for (int i = 0; i < numDetections; i++)
            {
                float confidence = detections[0, i, 4];
                if (confidence < confThreshold)
                    continue;

                // Extract bounding box in XYXY format (end2end model output)
                // These are in model input coordinates (960x960)
                float bx1 = detections[0, i, 0];
                float by1 = detections[0, i, 1];
                float bx2 = detections[0, i, 2];
                float by2 = detections[0, i, 3];
                int classId = (int)detections[0, i, 5];

                string className = (classId >= 0 && classId < _classNames.Length)
                    ? _classNames[classId]
                    : $"class_{classId}";

                Console.WriteLine($"[DEBUG] Detection {detIdx}: class={className}(id={classId}), conf={confidence:F4}, bbox_xyxy=({bx1:F1},{by1:F1},{bx2:F1},{by2:F1})");

                // Scale bbox to original image coordinates
                int ox1 = (int)(bx1 * scaleX);
                int oy1 = (int)(by1 * scaleY);
                int ox2 = (int)(bx2 * scaleX);
                int oy2 = (int)(by2 * scaleY);

                // Clamp to image boundaries
                ox1 = Math.Clamp(ox1, 0, origW);
                oy1 = Math.Clamp(oy1, 0, origH);
                ox2 = Math.Clamp(ox2, 0, origW);
                oy2 = Math.Clamp(oy2, 0, origH);

                int bw = ox2 - ox1;
                int bh = oy2 - oy1;

                Console.WriteLine($"[DEBUG]   bbox_orig=({ox1},{oy1},{ox2},{oy2}) size={bw}x{bh}");

                if (bw <= 0 || bh <= 0)
                {
                    Console.WriteLine($"[DEBUG]   SKIPPED: invalid bbox size");
                    continue;
                }

                // Extract 32 mask coefficients for this detection
                float[] maskCoeffs = new float[NumMaskCoeffs];
                for (int c = 0; c < NumMaskCoeffs; c++)
                {
                    maskCoeffs[c] = detections[0, i, 6 + c];
                }

                if (DebugOutputDir != null)
                {
                    Console.WriteLine($"[DEBUG]   maskCoeffs[0..4]: {string.Join(", ", maskCoeffs.Take(5).Select(v => v.ToString("F4")))}");
                }

                // Step 1: Compute mask = mask_coefficients @ protos => [240, 240]
                // This matches: masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
                using var maskMat = new Mat(MaskProtoH, MaskProtoW, MatType.CV_32FC1, Scalar.All(0));
                unsafe
                {
                    float* maskPtr = (float*)maskMat.Data;
                    for (int c = 0; c < NumMaskCoeffs; c++)
                    {
                        float coeff = maskCoeffs[c];
                        for (int my = 0; my < MaskProtoH; my++)
                        {
                            for (int mx = 0; mx < MaskProtoW; mx++)
                            {
                                maskPtr[my * MaskProtoW + mx] += coeff * protos[0, c, my, mx];
                            }
                        }
                    }
                }

                // Debug: save raw mask
                if (DebugOutputDir != null)
                {
                    Cv2.MinMaxLoc(maskMat, out double rawMin, out double rawMax);
                    Console.WriteLine($"[DEBUG]   raw mask 240x240 min={rawMin:F4}, max={rawMax:F4}");
                    using var rawVis = new Mat();
                    maskMat.ConvertTo(rawVis, MatType.CV_8UC1, 255.0 / (rawMax - rawMin + 1e-6), -rawMin * 255.0 / (rawMax - rawMin + 1e-6));
                    Cv2.ImWrite(Path.Combine(DebugOutputDir, $"01_raw_mask_240x240_det{detIdx}.png"), rawVis);
                }

                // Step 2: crop_mask — zero out everything outside bbox in mask proto space
                // This matches: crop_mask(masks, boxes=bboxes * ratios) where ratios = maskRatio
                int cmx1 = Math.Clamp((int)(bx1 * maskRatio), 0, MaskProtoW);
                int cmy1 = Math.Clamp((int)(by1 * maskRatio), 0, MaskProtoH);
                int cmx2 = Math.Clamp((int)(bx2 * maskRatio), 0, MaskProtoW);
                int cmy2 = Math.Clamp((int)(by2 * maskRatio), 0, MaskProtoH);

                // crop_mask: zero out rows above/below and cols left/right of bbox
                unsafe
                {
                    float* maskPtr = (float*)maskMat.Data;
                    for (int my = 0; my < MaskProtoH; my++)
                    {
                        for (int mx = 0; mx < MaskProtoW; mx++)
                        {
                            if (my < cmy1 || my >= cmy2 || mx < cmx1 || mx >= cmx2)
                            {
                                maskPtr[my * MaskProtoW + mx] = 0f;
                            }
                        }
                    }
                }

                Console.WriteLine($"[DEBUG]   crop_mask bbox in mask space: ({cmx1},{cmy1})-({cmx2},{cmy2})");

                // Debug: save cropped raw mask
                if (DebugOutputDir != null)
                {
                    Cv2.MinMaxLoc(maskMat, out double cropMin, out double cropMax);
                    Console.WriteLine($"[DEBUG]   cropped raw mask min={cropMin:F4}, max={cropMax:F4}");
                    using var cropVis = new Mat();
                    var visMax = Math.Max(Math.Abs(cropMin), Math.Abs(cropMax));
                    if (visMax > 0)
                        maskMat.ConvertTo(cropVis, MatType.CV_8UC1, 127.0 / visMax, 128.0);
                    else
                        maskMat.ConvertTo(cropVis, MatType.CV_8UC1, 1.0, 128.0);
                    Cv2.ImWrite(Path.Combine(DebugOutputDir, $"02_cropped_raw_mask_240x240_det{detIdx}.png"), cropVis);
                }

                // Step 3: Upsample to original image size via bilinear interpolation
                // YOLO upsamples to model input size (960x960), then scales boxes separately.  
                // We go directly to original size since we already scaled the boxes.
                using var fullMask = new Mat();
                Cv2.Resize(maskMat, fullMask, new Size(origW, origH), interpolation: InterpolationFlags.Linear);

                // Step 4: Threshold at > 0.0 (matches masks.gt_(0.0).byte())
                // This is equivalent to sigmoid > 0.5
                using var binaryMask = new Mat();
                Cv2.Threshold(fullMask, binaryMask, 0.0, 255.0, ThresholdTypes.Binary);

                // Convert to CV_8UC1
                using var mask8u = new Mat();
                binaryMask.ConvertTo(mask8u, MatType.CV_8UC1);

                // Debug: save binary mask
                if (DebugOutputDir != null)
                {
                    int nonZero = Cv2.CountNonZero(mask8u);
                    Console.WriteLine($"[DEBUG]   binary mask (gt 0.0) non-zero pixels: {nonZero} / {origW * origH}");
                    Cv2.ImWrite(Path.Combine(DebugOutputDir, $"03_binary_mask_det{detIdx}.png"), mask8u);
                }

                // Apply dilation for margin expansion (matching Python logic)
                Mat finalMask;
                if (marginBlockSize > 0)
                {
                    int marginPx = Math.Max(origW, origH) / marginBlockSize;
                    if (marginPx > 0)
                    {
                        int kernelSize = marginPx * 2 + 1;
                        Console.WriteLine($"[DEBUG]   dilation: marginPx={marginPx}, kernelSize={kernelSize}");
                        using var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Size(kernelSize, kernelSize));
                        finalMask = new Mat();
                        Cv2.Dilate(mask8u, finalMask, kernel, iterations: 1);
                    }
                    else
                    {
                        finalMask = mask8u.Clone();
                    }
                }
                else
                {
                    finalMask = mask8u.Clone();
                }

                // Debug: save final mask
                if (DebugOutputDir != null)
                {
                    int nonZeroFinal = Cv2.CountNonZero(finalMask);
                    Console.WriteLine($"[DEBUG]   final mask non-zero pixels: {nonZeroFinal} / {origW * origH}");
                    Cv2.ImWrite(Path.Combine(DebugOutputDir, $"04_final_mask_det{detIdx}.png"), finalMask);
                }

                // Filter: only keep detections with actual mask pixels (matches YOLO's keep = masks.amax > 0)
                if (Cv2.CountNonZero(finalMask) == 0)
                {
                    Console.WriteLine($"[DEBUG]   SKIPPED: empty mask after processing");
                    continue;
                }

                results.Add(new SegmentResult
                {
                    BoundingBox = new Rect(ox1, oy1, bw, bh),
                    ClassId = classId,
                    ClassName = className,
                    Confidence = confidence,
                    Mask = finalMask
                });

                detIdx++;
            }

            Console.WriteLine($"[DEBUG] Total results returned: {results.Count}");
            return results;
        }

        /// <summary>
        /// Applies element-wise sigmoid: 1 / (1 + exp(-x)) to a float Mat.
        /// </summary>
        private static void ApplySigmoid(Mat input, Mat output)
        {
            // Negate
            using var negated = new Mat();
            Cv2.Multiply(input, new Scalar(-1.0), negated);

            // Exp
            using var expMat = new Mat();
            Cv2.Exp(negated, expMat);

            // 1 + exp(-x)
            using var onePlusExp = new Mat();
            Cv2.Add(expMat, new Scalar(1.0), onePlusExp);

            // 1 / (1 + exp(-x))
            output.Create(input.Rows, input.Cols, input.Type());
            Cv2.Divide(new Scalar(1.0), onePlusExp, output);
        }

        /// <summary>
        /// Convenience method: creates a composite image with mosaic applied to detected segments.
        /// </summary>
        /// <param name="image">Original image (BGR).</param>
        /// <param name="results">Segmentation results from Predict().</param>
        /// <param name="blockSize">Mosaic block size divisor (higher = finer mosaic).</param>
        /// <param name="targetClasses">Class names to mosaic. If null, all classes are mosaiced.</param>
        /// <returns>New Mat with mosaic applied to detected regions.</returns>
        public static Mat ApplyMosaic(Mat image, List<SegmentResult> results, int blockSize = 100, string[]? targetClasses = null, string? debugOutputDir = null)
        {
            var output = image.Clone();
            int longestSide = Math.Max(image.Cols, image.Rows);
            int mosaicPixelSize = (int)Math.Ceiling((double)longestSide / blockSize);
            if (mosaicPixelSize < 1) mosaicPixelSize = 1;

            Console.WriteLine($"[DEBUG ApplyMosaic] image: {image.Cols}x{image.Rows}, channels={image.Channels()}, type={image.Type()}");
            Console.WriteLine($"[DEBUG ApplyMosaic] results count: {results.Count}, blockSize={blockSize}, mosaicPixelSize={mosaicPixelSize}");
            Console.WriteLine($"[DEBUG ApplyMosaic] targetClasses: {(targetClasses == null ? "null (all)" : string.Join(",", targetClasses))}");

            int applied = 0;
            foreach (var result in results)
            {
                Console.WriteLine($"[DEBUG ApplyMosaic] result: class={result.ClassName}, conf={result.Confidence:F4}, mask={result.Mask.Cols}x{result.Mask.Rows} type={result.Mask.Type()} nonzero={Cv2.CountNonZero(result.Mask)}");

                if (targetClasses != null && !targetClasses.Contains(result.ClassName))
                {
                    Console.WriteLine($"[DEBUG ApplyMosaic]   SKIPPED: class not in targetClasses");
                    continue;
                }

                // Generate mosaic: shrink then enlarge
                using var small = new Mat();
                Cv2.Resize(image, small,
                    new Size(image.Cols / mosaicPixelSize, image.Rows / mosaicPixelSize),
                    interpolation: InterpolationFlags.Linear);

                using var mosaiced = new Mat();
                Cv2.Resize(small, mosaiced,
                    new Size(image.Cols, image.Rows),
                    interpolation: InterpolationFlags.Nearest);

                // Debug: save mosaiced full image and the mask
                if (debugOutputDir != null)
                {
                    Directory.CreateDirectory(debugOutputDir);
                    Cv2.ImWrite(Path.Combine(debugOutputDir, $"10_mosaiced_full_{applied}.png"), mosaiced);
                    Cv2.ImWrite(Path.Combine(debugOutputDir, $"11_mask_for_copyto_{applied}.png"), result.Mask);
                }

                // Apply mosaic only in the mask region
                mosaiced.CopyTo(output, result.Mask);

                // Debug: save intermediate output after this mask is applied
                if (debugOutputDir != null)
                {
                    Cv2.ImWrite(Path.Combine(debugOutputDir, $"12_output_after_apply_{applied}.png"), output);
                }

                applied++;
            }

            Console.WriteLine($"[DEBUG ApplyMosaic] Applied mosaic to {applied} region(s)");
            return output;
        }

        /// <summary>
        /// Disposes of the ONNX inference session and related resources.
        /// </summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                _session.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }
}
