using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using AutoMosaicLib;
using OpenCvSharp;
using DragEventArgs = System.Windows.DragEventArgs;
using DataFormats = System.Windows.DataFormats;
using DragDropEffects = System.Windows.DragDropEffects;
using OpenFileDialog = Microsoft.Win32.OpenFileDialog;

namespace AutoMosaic
{
    public partial class MainWindow : System.Windows.Window
    {
        private static readonly string[] SupportedExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp" };
        private AppSettings _settings;
        private bool _isProcessing;

        public MainWindow()
        {
            InitializeComponent();
            _settings = AppSettings.Load();
        }

        // --- Settings ---

        private void Settings_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new SettingsWindow(_settings) { Owner = this };
            if (dlg.ShowDialog() == true)
                _settings = dlg.Settings;
        }

        // --- Drag & Drop ---

        private void Window_DragOver(object sender, DragEventArgs e)
        {
            if (e.Data.GetDataPresent(DataFormats.FileDrop))
            {
                e.Effects = _isProcessing ? DragDropEffects.None : DragDropEffects.Copy;
                DropZone.Tag = _isProcessing ? null : "hover";
            }
            else
            {
                e.Effects = DragDropEffects.None;
            }
            e.Handled = true;
        }

        private async void Window_Drop(object sender, DragEventArgs e)
        {
            DropZone.Tag = null;
            if (_isProcessing) return;
            if (!e.Data.GetDataPresent(DataFormats.FileDrop)) return;

            var paths = (string[])e.Data.GetData(DataFormats.FileDrop)!;
            if (paths.Length == 0) return;

            await ProcessDroppedPaths(paths);
        }

        private async void DropZone_Click(object sender, MouseButtonEventArgs e)
        {
            if (_isProcessing) return;

            var dlg = new OpenFileDialog
            {
                Title = "画像を選択",
                Filter = "画像ファイル|*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.webp|すべてのファイル|*.*",
                Multiselect = true,
                InitialDirectory = string.IsNullOrEmpty(_settings.LastInputPath)
                    ? Environment.GetFolderPath(Environment.SpecialFolder.MyPictures)
                    : Path.GetDirectoryName(_settings.LastInputPath) ?? ""
            };

            if (dlg.ShowDialog() == true && dlg.FileNames.Length > 0)
            {
                _settings.LastInputPath = dlg.FileNames[0];
                _settings.Save();
                await ProcessDroppedPaths(dlg.FileNames);
            }
        }

        // --- Processing ---

        private async Task ProcessDroppedPaths(string[] paths)
        {
            _isProcessing = true;
            ClearLog();

            try
            {
                // Collect all files grouped by source
                var files = new List<(string input, string basePath, bool fromDir)>();
                bool hasDirectory = false;

                foreach (var path in paths)
                {
                    if (File.Exists(path) && IsSupportedImage(path))
                    {
                        files.Add((path, Path.GetDirectoryName(path)!, false));
                    }
                    else if (Directory.Exists(path))
                    {
                        hasDirectory = true;
                        var dirFiles = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories)
                            .Where(IsSupportedImage)
                            .OrderBy(f => f);
                        foreach (var f in dirFiles)
                            files.Add((f, path, true));
                    }
                }

                if (files.Count == 0)
                {
                    Log("対応する画像ファイルが見つかりませんでした。");
                    return;
                }

                // Determine output directory
                string outputDir = DetermineOutputDirectory(paths[0], hasDirectory);
                if (string.IsNullOrEmpty(outputDir)) return;

                Log($"出力先: {outputDir}");
                Log($"処理対象: {files.Count} ファイル");

                // Resolve model path
                string modelPath = ResolveModelPath();
                if (!File.Exists(modelPath))
                {
                    Log($"❌ モデルが見つかりません: {modelPath}");
                    return;
                }

                var targets = _settings.GetTargetClasses();
                if (targets.Length == 0)
                {
                    Log("⚠ 対象クラスが選択されていません。設定を確認してください。");
                    return;
                }

                Log($"対象クラス: [{string.Join(", ", targets)}]");

                float conf = _settings.Confidence;
                int blockSize = _settings.BlockSize;
                bool useGpu = _settings.UseGpu;
                string filePrefix = _settings.FilePrefix;
                string fileSuffix = _settings.FileSuffix;
                string outputFormat = _settings.OutputFormat;
                int jpgQuality = _settings.JpgQuality;

                SetStatus("モデル読み込み中...");
                ProgressPanel.Visibility = Visibility.Visible;
                Progress.IsIndeterminate = true;
                TxtProgressDetail.Text = "モデル読み込み中...";
                TxtProgressPercent.Text = "";

                string? lastSavedFile = null;
                int successCount = 0;

                await Task.Run(() =>
                {
                    using var segmentator = new YoloSegmentator(modelPath, useGpu: useGpu);
                    Dispatcher.Invoke(() => Log($"モデル読み込み完了: [{string.Join(", ", segmentator.ClassNames)}]"));

                    for (int i = 0; i < files.Count; i++)
                    {
                        var (inputPath, basePath, fromDir) = files[i];
                        int idx = i;
                        Dispatcher.Invoke(() =>
                        {
                            SetStatus($"処理中: {Path.GetFileName(inputPath)}");
                            Progress.IsIndeterminate = false;
                            Progress.Maximum = files.Count;
                            Progress.Value = idx;
                            int pct = (int)((double)idx / files.Count * 100);
                            TxtProgressPercent.Text = $"{pct}%";
                            TxtProgressDetail.Text = $"{idx + 1} / {files.Count}: {Path.GetFileName(inputPath)}";
                        });

                        try
                        {
                            // Build output path
                            string outPath;
                            string ext = "." + outputFormat;
                            if (fromDir)
                            {
                                // Preserve directory structure
                                string relativePath = Path.GetRelativePath(basePath, inputPath);
                                string relDir = Path.GetDirectoryName(relativePath) ?? "";
                                string baseName = Path.GetFileNameWithoutExtension(relativePath);
                                string outName = $"{filePrefix}{baseName}{fileSuffix}{ext}";
                                outPath = Path.Combine(outputDir, relDir, outName);
                            }
                            else
                            {
                                // Single file
                                string baseName = Path.GetFileNameWithoutExtension(inputPath);
                                string outName = $"{filePrefix}{baseName}{fileSuffix}{ext}";
                                outPath = Path.Combine(outputDir, outName);
                            }

                            Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);

                            var image = Cv2.ImRead(inputPath);
                            if (image.Empty())
                            {
                                Dispatcher.Invoke(() => Log($"  ⚠ 読み込み失敗: {Path.GetFileName(inputPath)}"));
                                continue;
                            }

                            var results = segmentator.Predict(image, confThreshold: conf, marginBlockSize: _settings.MarginBlockSize);
                            Dispatcher.Invoke(() => Log($"  [{idx + 1}/{files.Count}] {Path.GetFileName(inputPath)} → {results.Count} 検出"));

                            using var output = YoloSegmentator.ApplyMosaic(image, results, blockSize: blockSize, targetClasses: targets);

                            // Save with format-specific params
                            if (outputFormat == "jpg")
                                Cv2.ImWrite(outPath, output, new ImageEncodingParam(ImwriteFlags.JpegQuality, jpgQuality));
                            else if (outputFormat == "webp")
                                Cv2.ImWrite(outPath, output, new ImageEncodingParam(ImwriteFlags.WebPQuality, jpgQuality));
                            else
                                Cv2.ImWrite(outPath, output);

                            lastSavedFile = outPath;
                            successCount++;

                            foreach (var r in results) r.Mask.Dispose();
                            image.Dispose();
                        }
                        catch (Exception ex)
                        {
                            Dispatcher.Invoke(() => Log($"  ❌ {Path.GetFileName(inputPath)}: {ex.Message}"));
                        }
                    }
                });

                Progress.Value = files.Count;
                TxtProgressPercent.Text = "100%";
                TxtProgressDetail.Text = $"完了: {successCount}/{files.Count} ファイル";
                Log($"\n✅ 完了: {successCount}/{files.Count} ファイル → {outputDir}");
                SetStatus("完了");

                _settings.LastOutputPath = outputDir;
                _settings.Save();

                // Open folder
                if (_settings.OpenAfterSave && _settings.FixedOutputPath && lastSavedFile != null)
                {
                    if (files.Count == 1 && File.Exists(lastSavedFile))
                        Process.Start("explorer", $"/select, \"{lastSavedFile}\"");
                    else
                        Process.Start("explorer", $"\"{outputDir}\"");
                }
            }
            catch (Exception ex)
            {
                Log($"\n❌ 致命的エラー: {ex.Message}");
                SetStatus("エラー");
            }
            finally
            {
                _isProcessing = false;
                // Keep progress visible for a moment, then hide after 3 seconds
                _ = Task.Delay(3000).ContinueWith(_ => Dispatcher.Invoke(() =>
                {
                    if (!_isProcessing) ProgressPanel.Visibility = Visibility.Collapsed;
                }));
            }
        }

        /// <summary>
        /// Determines the output directory.
        /// - Fixed path: use settings.
        /// - Directory input: create sibling directory with FolderSuffix (e.g. "input_mosaic").
        /// - File input: ask user via dialog.
        /// </summary>
        private string DetermineOutputDirectory(string firstPath, bool hasDirectory)
        {
            // Fixed output path
            if (_settings.FixedOutputPath && !string.IsNullOrWhiteSpace(_settings.LastOutputPath))
                return _settings.LastOutputPath;

            // Directory dropped → auto-create sibling dir with suffix
            if (hasDirectory && Directory.Exists(firstPath))
            {
                string dirName = Path.GetFileName(firstPath.TrimEnd(Path.DirectorySeparatorChar));
                string parentDir = Path.GetDirectoryName(firstPath.TrimEnd(Path.DirectorySeparatorChar)) ?? firstPath;
                string outputDir = Path.Combine(parentDir, dirName + _settings.FolderSuffix);
                return outputDir;
            }

            // Single file(s) → ask user
            var dlg = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "保存先フォルダを選択",
                UseDescriptionForTitle = true,
                InitialDirectory = string.IsNullOrEmpty(_settings.LastOutputPath)
                    ? Path.GetDirectoryName(firstPath) ?? ""
                    : _settings.LastOutputPath
            };

            return dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK ? dlg.SelectedPath : "";
        }

        private string ResolveModelPath()
        {
            string modelPath = _settings.ModelPath;
            if (Path.IsPathRooted(modelPath)) return modelPath;

            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            string fullPath = Path.Combine(exeDir, modelPath);
            return File.Exists(fullPath) ? fullPath : modelPath;
        }

        // --- UI helpers ---

        private void Log(string message)
        {
            if (TxtLog.Text == "画像をドロップして開始...")
                TxtLog.Text = message;
            else
                TxtLog.Text += "\n" + message;
            LogScroll.ScrollToEnd();
        }

        private void ClearLog() => TxtLog.Text = "";
        private void SetStatus(string status) => TxtStatus.Text = status;
        private static bool IsSupportedImage(string path) =>
            SupportedExtensions.Contains(Path.GetExtension(path).ToLowerInvariant());
    }
}