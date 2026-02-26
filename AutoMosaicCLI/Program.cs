using System;
using System.IO;
using System.Linq;
using AutoMosaicLib;
using OpenCvSharp;

namespace AutoMosaicCLI;

class Program
{
    private static readonly string[] SupportedExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp" };

    static int Main(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            PrintUsage();
            return 0;
        }

        // Parse arguments
        string? inputPath = null;
        string? outputPath = null;
        string modelPath = "sensitive_detect_v06.onnx";
        float confidence = 0.5f;
        int blockSize = 100;
        int marginBlockSize = 100;
        string targetClasses = "pussy,penis";
        bool recursive = false;
        bool useGpu = false;
        string? debugDir = null;
        string outputSuffix = "_mosaic";
        string outputFormat = "png";

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "-i":
                case "--input":
                    inputPath = GetNextArg(args, ref i);
                    break;
                case "-o":
                case "--output":
                    outputPath = GetNextArg(args, ref i);
                    break;
                case "-m":
                case "--model":
                    modelPath = GetNextArg(args, ref i);
                    break;
                case "-c":
                case "--conf":
                    confidence = float.Parse(GetNextArg(args, ref i));
                    break;
                case "-b":
                case "--block-size":
                    blockSize = int.Parse(GetNextArg(args, ref i));
                    break;
                case "--margin":
                    marginBlockSize = int.Parse(GetNextArg(args, ref i));
                    break;
                case "-t":
                case "--target":
                    targetClasses = GetNextArg(args, ref i);
                    break;
                case "-r":
                case "--recursive":
                    recursive = true;
                    break;
                case "--gpu":
                    useGpu = true;
                    break;
                case "--debug":
                    debugDir = GetNextArg(args, ref i);
                    break;
                case "--suffix":
                    outputSuffix = GetNextArg(args, ref i);
                    break;
                case "--format":
                    outputFormat = GetNextArg(args, ref i).TrimStart('.');
                    break;
                default:
                    if (inputPath == null && !args[i].StartsWith("-"))
                        inputPath = args[i];
                    else
                        Console.Error.WriteLine($"Warning: Unknown argument '{args[i]}'");
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Error: Input path is required. Use -i <path> or pass it as the first argument.");
            return 1;
        }

        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine($"Error: Model file not found: {modelPath}");
            return 1;
        }

        var targets = targetClasses.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        // Initialize model
        Console.WriteLine($"Loading model: {modelPath} (GPU: {useGpu})");
        using var segmentator = new YoloSegmentator(modelPath, useGpu: useGpu);
        segmentator.DebugOutputDir = debugDir;
        Console.WriteLine($"Model loaded. Classes: [{string.Join(", ", segmentator.ClassNames)}]");
        Console.WriteLine($"Target classes: [{string.Join(", ", targets)}]");
        Console.WriteLine($"Confidence: {confidence}, Block size: {blockSize}, Margin: {marginBlockSize}");

        bool isInputFile = File.Exists(inputPath);
        bool isInputDir = Directory.Exists(inputPath);

        if (!isInputFile && !isInputDir)
        {
            Console.Error.WriteLine($"Error: Input path not found: {inputPath}");
            return 1;
        }

        if (isInputFile)
        {
            // Single file mode
            string outPath = outputPath ?? GenerateOutputPath(inputPath, outputSuffix, outputFormat);
            return ProcessFile(segmentator, inputPath, outPath, confidence, blockSize, marginBlockSize, targets, debugDir) ? 0 : 1;
        }
        else
        {
            // Directory mode
            string outDir = outputPath ?? Path.Combine(inputPath, "output");
            return ProcessDirectory(segmentator, inputPath, outDir, confidence, blockSize, marginBlockSize, targets, recursive, outputSuffix, outputFormat, debugDir);
        }
    }

    static bool ProcessFile(
        YoloSegmentator segmentator, string inputPath, string outputPath,
        float confidence, int blockSize, int marginBlockSize, string[] targets, string? debugDir)
    {
        Console.WriteLine($"\nProcessing: {inputPath}");

        var image = Cv2.ImRead(inputPath);
        if (image.Empty())
        {
            Console.Error.WriteLine($"  Error: Failed to load image: {inputPath}");
            return false;
        }

        Console.WriteLine($"  Image size: {image.Cols}x{image.Rows}");

        var results = segmentator.Predict(image, confThreshold: confidence, marginBlockSize: marginBlockSize);
        Console.WriteLine($"  Detections: {results.Count}");

        foreach (var r in results)
        {
            Console.WriteLine($"    {r.ClassName} (conf={r.Confidence:F3}) bbox=({r.BoundingBox.X},{r.BoundingBox.Y},{r.BoundingBox.Width}x{r.BoundingBox.Height})");
        }

        using var output = YoloSegmentator.ApplyMosaic(image, results, blockSize: blockSize, targetClasses: targets, debugOutputDir: debugDir);

        // Ensure output directory exists
        var outDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outDir))
            Directory.CreateDirectory(outDir);

        Cv2.ImWrite(outputPath, output);
        Console.WriteLine($"  Saved: {outputPath}");

        // Cleanup
        foreach (var r in results) r.Mask.Dispose();
        image.Dispose();

        return true;
    }

    static int ProcessDirectory(
        YoloSegmentator segmentator, string inputDir, string outputDir,
        float confidence, int blockSize, int marginBlockSize, string[] targets,
        bool recursive, string outputSuffix, string outputFormat, string? debugDir)
    {
        var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        var files = Directory.GetFiles(inputDir, "*.*", searchOption)
            .Where(f => SupportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f)
            .ToList();

        Console.WriteLine($"\nFound {files.Count} image(s) in: {inputDir} (recursive: {recursive})");
        Console.WriteLine($"Output directory: {outputDir}");

        int success = 0;
        int failed = 0;

        for (int i = 0; i < files.Count; i++)
        {
            var file = files[i];
            Console.WriteLine($"\n[{i + 1}/{files.Count}] {file}");

            // Preserve relative directory structure
            string relativePath = Path.GetRelativePath(inputDir, file);
            string relativeDir = Path.GetDirectoryName(relativePath) ?? "";
            string baseName = Path.GetFileNameWithoutExtension(relativePath);
            string outPath = Path.Combine(outputDir, relativeDir, $"{baseName}{outputSuffix}.{outputFormat}");

            if (ProcessFile(segmentator, file, outPath, confidence, blockSize, marginBlockSize, targets, debugDir))
                success++;
            else
                failed++;
        }

        Console.WriteLine($"\n=== Done ===");
        Console.WriteLine($"  Success: {success}, Failed: {failed}, Total: {files.Count}");
        return failed > 0 ? 1 : 0;
    }

    static string GenerateOutputPath(string inputPath, string suffix, string format)
    {
        string dir = Path.GetDirectoryName(inputPath) ?? ".";
        string baseName = Path.GetFileNameWithoutExtension(inputPath);
        return Path.Combine(dir, $"{baseName}{suffix}.{format}");
    }

    static string GetNextArg(string[] args, ref int index)
    {
        index++;
        if (index >= args.Length)
            throw new ArgumentException($"Missing value for argument: {args[index - 1]}");
        return args[index];
    }

    static void PrintUsage()
    {
        Console.WriteLine(@"
AutoMosaic CLI - Segmentation Mosaic Tool

USAGE:
  AutoMosaicCLI [options] <input>
  AutoMosaicCLI -i <image_or_directory> [-o <output>] [options]

INPUT:
  -i, --input <path>     Input image file or directory (required)

OUTPUT:
  -o, --output <path>    Output file path (single file) or directory (batch mode)
                         Default: <input_name>_mosaic.png (file) or <input>/output/ (dir)
  --suffix <text>        Output filename suffix (default: _mosaic)
  --format <ext>         Output format: png, jpg, bmp, webp (default: png)

MODEL:
  -m, --model <path>     Path to ONNX model file (default: sd.onnx)
  --gpu                  Use GPU (CUDA) for inference

DETECTION:
  -c, --conf <float>     Confidence threshold 0.0-1.0 (default: 0.5)
  -t, --target <classes> Comma-separated target classes to mosaic (default: pussy,penis)

MOSAIC:
  -b, --block-size <n>   Mosaic block size divisor - higher = finer (default: 100)
  --margin <n>           Mask dilation margin divisor - higher = smaller margin (default: 100)

BATCH:
  -r, --recursive        Process subdirectories recursively

DEBUG:
  --debug <dir>          Save debug images to specified directory

EXAMPLES:
  # Single image
  AutoMosaicCLI -i photo.jpg -o result.png

  # With custom confidence and block size
  AutoMosaicCLI -i photo.jpg -c 0.3 -b 50

  # Batch process a directory
  AutoMosaicCLI -i ./input_images -o ./output_images -r

  # Use GPU with debug output
  AutoMosaicCLI -i photo.jpg --gpu --debug ./debug_output
");
    }
}
