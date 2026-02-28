using System.Windows;
using OpenFileDialog = Microsoft.Win32.OpenFileDialog;

namespace AutoMosaic
{
    public partial class SettingsWindow : Window
    {
        public AppSettings Settings { get; private set; }
        public bool Saved { get; private set; }

        public SettingsWindow(AppSettings settings)
        {
            InitializeComponent();
            Settings = settings;
            LoadFromSettings();
        }

        private void LoadFromSettings()
        {
            TxtModel.Text = Settings.ModelPath;
            SliderConf.Value = Settings.Confidence;
            SliderBlock.Value = Settings.BlockSize;
            SliderExpand.Value = Settings.ExpandRatio;

            ChkPussy.IsChecked = Settings.TargetPussy;
            ChkAnus.IsChecked = Settings.TargetAnus;
            ChkPenis.IsChecked = Settings.TargetPenis;
            ChkNipple.IsChecked = Settings.TargetNipple;

            TxtFilePrefix.Text = Settings.FilePrefix;
            TxtFileSuffix.Text = Settings.FileSuffix;
            TxtFolderSuffix.Text = Settings.FolderSuffix;

            // Select format in ComboBox
            foreach (System.Windows.Controls.ComboBoxItem item in CmbFormat.Items)
            {
                if ((string)item.Tag == Settings.OutputFormat)
                {
                    CmbFormat.SelectedItem = item;
                    break;
                }
            }
            if (CmbFormat.SelectedIndex < 0) CmbFormat.SelectedIndex = 0;

            SliderQuality.Value = Settings.JpgQuality;

            TxtOutputPath.Text = Settings.LastOutputPath;
            ChkFixedOutput.IsChecked = Settings.FixedOutputPath;
            ChkOpenAfterSave.IsChecked = Settings.OpenAfterSave;
            ChkGpu.IsChecked = Settings.UseGpu;

            // Select overwrite mode
            string modeTag = Settings.OverwriteMode.ToString();
            foreach (System.Windows.Controls.ComboBoxItem item in CmbOverwrite.Items)
            {
                if ((string)item.Tag == modeTag)
                {
                    CmbOverwrite.SelectedItem = item;
                    break;
                }
            }
            if (CmbOverwrite.SelectedIndex < 0) CmbOverwrite.SelectedIndex = 0;
        }

        private void SaveToSettings()
        {
            Settings.ModelPath = TxtModel.Text;
            Settings.Confidence = (float)SliderConf.Value;
            Settings.BlockSize = (int)SliderBlock.Value;
            Settings.ExpandRatio = (float)SliderExpand.Value;

            Settings.TargetPussy = ChkPussy.IsChecked == true;
            Settings.TargetAnus = ChkAnus.IsChecked == true;
            Settings.TargetPenis = ChkPenis.IsChecked == true;
            Settings.TargetNipple = ChkNipple.IsChecked == true;

            Settings.FilePrefix = TxtFilePrefix.Text;
            Settings.FileSuffix = TxtFileSuffix.Text;
            Settings.FolderSuffix = TxtFolderSuffix.Text;

            Settings.OutputFormat = (CmbFormat.SelectedItem is System.Windows.Controls.ComboBoxItem sel)
                ? (string)sel.Tag : "png";
            Settings.JpgQuality = (int)SliderQuality.Value;

            Settings.LastOutputPath = TxtOutputPath.Text;
            Settings.FixedOutputPath = ChkFixedOutput.IsChecked == true;
            Settings.OpenAfterSave = ChkOpenAfterSave.IsChecked == true;
            Settings.UseGpu = ChkGpu.IsChecked == true;

            Settings.OverwriteMode = (CmbOverwrite.SelectedItem is System.Windows.Controls.ComboBoxItem owSel
                && int.TryParse((string)owSel.Tag, out int mode)) ? mode : 0;
        }

        private void SliderConf_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtConfValue != null) TxtConfValue.Text = e.NewValue.ToString("F2");
        }

        private void SliderBlock_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtBlockValue != null) TxtBlockValue.Text = ((int)e.NewValue).ToString();
        }

        private void SliderExpand_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtExpandValue != null) TxtExpandValue.Text = ((int)e.NewValue).ToString();
        }

        private void SliderQuality_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtQualityValue != null) TxtQualityValue.Text = ((int)e.NewValue).ToString();
        }

        private void BrowseModel_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Title = "ONNX モデルを選択",
                Filter = "ONNX モデル|*.onnx|すべてのファイル|*.*"
            };
            if (dlg.ShowDialog() == true) TxtModel.Text = dlg.FileName;
        }

        private void BrowseOutput_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new System.Windows.Forms.FolderBrowserDialog
            {
                Description = "出力先フォルダを選択",
                UseDescriptionForTitle = true
            };
            if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                TxtOutputPath.Text = dlg.SelectedPath;
        }

        private void Save_Click(object sender, RoutedEventArgs e)
        {
            SaveToSettings();
            Settings.Save();
            Saved = true;
            DialogResult = true;
            Close();
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
