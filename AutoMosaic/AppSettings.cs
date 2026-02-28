using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace AutoMosaic
{
    public class AppSettings
    {
        public string LastInputPath { get; set; } = "";
        public string LastOutputPath { get; set; } = "";
        public string ModelPath { get; set; } = "sensitive_detect_v06.onnx";
        public float Confidence { get; set; } = 0.5f;
        public int BlockSize { get; set; } = 100;
        public int MarginBlockSize { get; set; } = 100;
        public float ExpandRatio { get; set; } = 0f;
        public bool UseGpu { get; set; } = false;
        public bool FixedOutputPath { get; set; } = false;
        public bool OpenAfterSave { get; set; } = true;

        // Target classes as individual toggles
        public bool TargetPussy { get; set; } = true;
        public bool TargetAnus { get; set; } = false;
        public bool TargetPenis { get; set; } = true;
        public bool TargetNipple { get; set; } = false;

        // Output naming
        public string FilePrefix { get; set; } = "";
        public string FileSuffix { get; set; } = "_mosaic";
        public string FolderSuffix { get; set; } = "_mosaic";

        // Output format
        public string OutputFormat { get; set; } = "png";  // png, jpg, bmp, webp
        public int JpgQuality { get; set; } = 95;  // 1-100

        // File conflict handling: 0=AutoRename, 1=Overwrite, 2=ConfirmDialog
        public int OverwriteMode { get; set; } = 0;

        /// <summary>
        /// Returns selected target class names as a string array.
        /// </summary>
        public string[] GetTargetClasses()
        {
            var list = new List<string>();
            if (TargetPussy) list.Add("pussy");
            if (TargetAnus) list.Add("anus");
            if (TargetPenis) list.Add("penis");
            if (TargetNipple) list.Add("nipple");
            return list.ToArray();
        }

        private static readonly string SettingsPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "AutoMosaic", "settings.json");

        public void Save()
        {
            var dir = Path.GetDirectoryName(SettingsPath)!;
            Directory.CreateDirectory(dir);
            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(SettingsPath, json);
        }

        public static AppSettings Load()
        {
            try
            {
                if (File.Exists(SettingsPath))
                {
                    var json = File.ReadAllText(SettingsPath);
                    return JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
                }
            }
            catch { }
            return new AppSettings();
        }
    }
}
