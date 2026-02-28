using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using OpenCvSharp;

namespace AutoMosaicLib
{
    public class VideoProcessor
    {
        private readonly YoloSegmentator _segmentator;
        private readonly int _blockSize;
        private readonly float _confidence;
        private readonly int _marginBlockSize;
        private readonly float _expandRatio;
        private readonly string[]? _targetClasses;
        private readonly Action<int, int>? _onProgress;

        // Optical Flow tracking variables
        private readonly int _maxFlowFrames = 10;
        private Mat? _prevGraySmall;
        private Mat? _prevMaskFull;
        private int _missingFramesCount = 0;

        public VideoProcessor(
            YoloSegmentator segmentator, 
            int blockSize = 100, 
            float confidence = 0.5f, 
            int marginBlockSize = 100,
            float expandRatio = 0f,
            string[]? targetClasses = null,
            Action<int, int>? onProgress = null)
        {
            _segmentator = segmentator ?? throw new ArgumentNullException(nameof(segmentator));
            _blockSize = blockSize;
            _confidence = confidence;
            _marginBlockSize = marginBlockSize;
            _expandRatio = expandRatio;
            _targetClasses = targetClasses;
            _onProgress = onProgress;
        }

        public async Task ProcessVideoAsync(string inputPath, string outputPath)
        {
            if (!File.Exists(inputPath))
                throw new FileNotFoundException("Input video file not found.", inputPath);

            // 1. Get video info (resolution, fps, total frames) using ffprobe
            var (width, height, fps, totalFrames) = await GetVideoInfoAsync(inputPath);
            if (width <= 0 || height <= 0)
                throw new InvalidOperationException("Could not determine video resolution.");

            int frameSize = width * height * 3; // BGR24

            // 2. Setup FFmpeg input process (decode to raw BGR24 pipe)
            var startInfoIn = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = $"-v error -i \"{inputPath}\" -map 0:v:0 -f rawvideo -pix_fmt bgr24 -",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            // 3. Setup FFmpeg output process (encode raw BGR24 pipe to MP4, copy audio)
            var startInfoOut = new ProcessStartInfo
            {
                FileName = "ffmpeg",
                Arguments = $"-y -f rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - -i \"{inputPath}\" -map 0:v:0 -map 1:a:0? -c:a copy -c:v libx264 -pix_fmt yuv420p -preset fast -crf 23 -shortest \"{outputPath}\"",
                RedirectStandardInput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            var processIn = Process.Start(startInfoIn);
            if (processIn == null) throw new InvalidOperationException("Failed to start ffmpeg input process.");

            var processOut = Process.Start(startInfoOut);
            if (processOut == null) throw new InvalidOperationException("Failed to start ffmpeg output process.");

            try
            {
                using var inStream = processIn.StandardOutput.BaseStream;
                using var outStream = processOut.StandardInput.BaseStream;

                byte[] frameBuffer = new byte[frameSize];
                int frameIndex = 0;

                while (true)
                {
                    // Read exactly one frame from FFmpeg stdout
                    int bytesRead = 0;
                    while (bytesRead < frameSize)
                    {
                        int read = await inStream.ReadAsync(frameBuffer.AsMemory(bytesRead, frameSize - bytesRead));
                        if (read == 0) break; // EOF
                        bytesRead += read;
                    }

                    if (bytesRead < frameSize)
                        break; // Incomplete frame / End of stream

                    // Create Mat from byte array (does not copy data)
                    unsafe
                    {
                        fixed (byte* p = frameBuffer)
                        {
                            using var frame = Mat.FromPixelData(height, width, MatType.CV_8UC3, (IntPtr)p);

                            // 1. Process frame with YOLO
                            var results = _segmentator.Predict(frame, confThreshold: _confidence, marginBlockSize: _marginBlockSize, expandRatio: _expandRatio);
                            
                            // 2. Build current mask from YOLO detections
                            using var currentBinaryMask = new Mat(height, width, MatType.CV_8UC1, Scalar.All(0));
                            bool hasYoloDetections = false;

                            if (results.Count > 0)
                            {
                                foreach (var r in results)
                                {
                                    if (_targetClasses == null || _targetClasses.Contains(r.ClassName))
                                    {
                                        Cv2.BitwiseOr(currentBinaryMask, r.Mask, currentBinaryMask);
                                        hasYoloDetections = true;
                                    }
                                }
                            }

                            using var finalMask = new Mat();
                            
                            // 3. Prepare grayscale downscaled frame for Optical Flow
                            using var currGrayFull = new Mat();
                            Cv2.CvtColor(frame, currGrayFull, ColorConversionCodes.BGR2GRAY);
                            using var currGraySmall = new Mat();
                            int flowWidth = 640;
                            int flowHeight = (int)(height * ((double)flowWidth / width));
                            Cv2.Resize(currGrayFull, currGraySmall, new Size(flowWidth, flowHeight));

                            if (hasYoloDetections)
                            {
                                // Object found: use YOLO mask and update tracking baselines
                                currentBinaryMask.CopyTo(finalMask);
                                
                                if (_prevGraySmall != null) _prevGraySmall.Dispose();
                                _prevGraySmall = currGraySmall.Clone();
                                
                                if (_prevMaskFull != null) _prevMaskFull.Dispose();
                                _prevMaskFull = currentBinaryMask.Clone();
                                
                                _missingFramesCount = 0;
                            }
                            else
                            {
                                // Object NOT found: try Optical Flow interpolation
                                if (_prevGraySmall != null && _prevMaskFull != null && _missingFramesCount < _maxFlowFrames)
                                {
                                    // Calculate Dense Optical Flow (Farneback)
                                    using var flow = new Mat();
                                    Cv2.CalcOpticalFlowFarneback(_prevGraySmall, currGraySmall, flow, 
                                        pyrScale: 0.5, levels: 3, winsize: 15, iterations: 3, polyN: 5, polySigma: 1.2, flags: 0);
                                    
                                    // Scale flow vectors up to full resolution
                                    using var flowFull = new Mat();
                                    Cv2.Resize(flow, flowFull, new Size(width, height));
                                    
                                    float scaleX = (float)width / flowWidth;
                                    float scaleY = (float)height / flowHeight;
                                    Cv2.Multiply(flowFull, new Scalar(scaleX, scaleY), flowFull);

                                    // Build remap maps from flow
                                    using var mapX = new Mat(height, width, MatType.CV_32FC1);
                                    using var mapY = new Mat(height, width, MatType.CV_32FC1);
                                    
                                    unsafe
                                    {
                                        float* flowPtr = (float*)flowFull.Data;
                                        float* mxPtr = (float*)mapX.Data;
                                        float* myPtr = (float*)mapY.Data;
                                        int totalPixels = width * height;
                                        for (int i = 0; i < totalPixels; i++)
                                        {
                                            int y = i / width;
                                            int x = i % width;
                                            // OpenCV flow: index 0 is DX (x displacement), index 1 is DY (y displacement)
                                            // remap expects map to contain the SOURCE coordinate for each DESTINATION pixel.
                                            // flow(x,y) is the displacement from prev to curr.
                                            // So source coordinate in prev is: curr_x - dx, curr_y - dy
                                            mxPtr[i] = x - flowPtr[i * 2];
                                            myPtr[i] = y - flowPtr[i * 2 + 1];
                                        }
                                    }

                                    // Warp the previous mask using the calculated maps
                                    using var warpedMask = new Mat();
                                    Cv2.Remap(_prevMaskFull, warpedMask, mapX, mapY, InterpolationFlags.Nearest);

                                    // Use warped mask for current frame
                                    warpedMask.CopyTo(finalMask);

                                    // Update baselines for next frame (chains the tracking)
                                    _prevGraySmall.Dispose();
                                    _prevGraySmall = currGraySmall.Clone();

                                    _prevMaskFull.Dispose();
                                    _prevMaskFull = warpedMask.Clone();

                                    _missingFramesCount++;
                                    Console.WriteLine($"[DEBUG] Frame {frameIndex}: Optical Flow interpolation applied (missing {_missingFramesCount} frames)");
                                }
                                else
                                {
                                    // Unrecoverable missing object, leave mask empty
                                    currentBinaryMask.CopyTo(finalMask);
                                    // Do not reset _missingFramesCount here to avoid permanently freezing the old mask if it's completely lost
                                }
                            }

                            // 4. Apply Mosaic using the final mask
                            if (Cv2.CountNonZero(finalMask) > 0)
                            {
                                using var mosaicedFrame = YoloSegmentator.ApplyMosaic(frame, finalMask, blockSize: _blockSize);
                                
                                // Copy processed data back to buffer
                                int copySize = (int)(mosaicedFrame.Total() * mosaicedFrame.ElemSize());
                                System.Runtime.InteropServices.Marshal.Copy(mosaicedFrame.Data, frameBuffer, 0, copySize);
                            }
                            
                            // Dispose memory
                            foreach (var r in results) r.Mask.Dispose();
                        }
                    }

                    // Write frame to output FFmpeg stdin
                    await outStream.WriteAsync(frameBuffer.AsMemory(0, frameSize));
                    
                    frameIndex++;
                    _onProgress?.Invoke(frameIndex, totalFrames);
                }

                // Close stdin to signal EOF to output FFmpeg
                outStream.Close();
                
                // Wait for processes to exit
                await Task.WhenAll(
                    processIn.WaitForExitAsync(),
                    processOut.WaitForExitAsync()
                );
            }
            finally
            {
                if (!processIn.HasExited) processIn.Kill();
                if (!processOut.HasExited) processOut.Kill();
                processIn.Dispose();
                processOut.Dispose();

                _prevGraySmall?.Dispose();
                _prevMaskFull?.Dispose();
            }
        }

        private async Task<(int width, int height, double fps, int totalFrames)> GetVideoInfoAsync(string inputPath)
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = "ffprobe",
                Arguments = $"-v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=noprint_wrappers=1:nokey=1 \"{inputPath}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process == null) return (0, 0, 0, 0);

            string output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();

            var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (lines.Length < 3) return (0, 0, 0, 0);

            int width = int.Parse(lines[0]);
            int height = int.Parse(lines[1]);
            
            // Parse fraction fps (e.g. 30000/1001)
            double fps = 30.0;
            var fpsParts = lines[2].Split('/');
            if (fpsParts.Length == 2 && double.TryParse(fpsParts[0], out double num) && double.TryParse(fpsParts[1], out double den))
            {
                fps = num / den;
            }
            else if (double.TryParse(lines[2], out double f))
            {
                fps = f;
            }

            // frames could be N/A
            int totalFrames = 0;
            if (lines.Length >= 4 && int.TryParse(lines[3], out int tf))
            {
                totalFrames = tf;
            }

            return (width, height, Math.Round(fps, 3), totalFrames);
        }
    }
}
