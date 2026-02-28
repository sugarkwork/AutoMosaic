# YOLO モデルの ONNX 変換ガイド (AutoMosaic 用)

このドキュメントでは、AutoMosaic で使用するための YOLO26 セグメンテーションモデル（.pt）を ONNX 形式（.onnx）に変換する際の重要な基準とパラメータについて説明します。

## 重要なポイント：`end2end=False`

**結論から言うと、AutoMosaic で使用する ONNX モデルは必ず `end2end=False` でエクスポートする必要があります。**

### なぜ `end2end=False` なのか？
最近の Ultralytics の標準エクスポート（`end2end=True` または未指定）では、ONNX モデルの内部に **NMS（Non-Maximum Suppression: 非最大値抑制）** の処理が焼き込まれます。

NMS が組み込まれたモデルは推論結果がすぐに使える形（例: `[1, 300, 38]`）になるため便利ですが、以下の大きな問題があります。
- エクスポート時のデフォルトの「信頼度（Confidence）」や「IoU」の閾値がモデル内部に固定されてしまいます。
- その結果、AutoMosaic の画面上で Conf スライダーを `0.1` のように低く設定しても、**モデル内部で既に 0.25 以下の検出が切り捨てられているため、低信頼度の対象物が検出漏れ**してしまいます。

`end2end=False` を指定することで、モデルは全ての生の検出結果（例: 18,900件のバウンディングボックス候補：`[1, 40, 18900]`）を出力するようになります。これにより、C# プログラム側で自由に閾値を設定し、確実にフィルタリングを行うことができます。

## 変換コマンド

Python 環境で Ultralytics をインストールした上で、以下のコマンドを実行します。

### CLI の場合
```bash
yolo export model=sensitive_detect_v06.pt format=onnx imgsz=960 simplify=True opset=17 end2end=False
```

### Python スクリプトの場合
```python
from ultralytics import YOLO

# 1. トレーニング済みの YOLO モデルを読み込む
model = YOLO('sensitive_detect_v06.pt')

# 2. 推奨パラメータで ONNX にエクスポート
model.export(
    format='onnx',       # 出力フォーマットを ONNX に指定
    imgsz=960,           # 入力画像サイズ（学習時の imgsz=960 と合わせる）
    simplify=True,       # ONNX グラフの最適化・簡略化を行う（ロードや推論が高速になる）
    opset=17,            # ONNX の Opset バージョン（ONNX Runtime との互換性のために 17 が推奨）
    end2end=False        # ⚠️ NMS を組み込まない（C# 側で全検出結果から閾値処理するため必須）
)
```

## 各パラメータの解説

| パラメータ | 設定値 | 理由・目的 |
|---|---|---|
| `format` | `'onnx'` | C# (ONNX Runtime) で推論を実行するため。 |
| `imgsz` | `960` | AutoMosaic の学習時パラメータ（`imgsz=960`）に合わせる必要があります。入力解像度が高いほうが小さな対象物を検出しやすくなります。 |
| `simplify` | `True` | モデル構造の冗長な部分を最適化します。C# での推論速度が向上します。 |
| `opset` | `17` | 最新機能のサポートと C# ONNX Runtime との互換性のバランスが良いバージョンです。 （エラーが出る場合は `13` 等に下げてください） |
| `end2end` | `False` | 検出漏れを防ぐための最も重要なオプション。NMS処理を外部（C#側）に委ねます。 |

## C# 側の処理について（参考）
`end2end=False` で出力された ONNX は生データ（`[1, 40, 18900]` のようなテンソル）を返すため、AutoMosaic の [YoloSegmentator.cs](../AutoMosaicLib/YoloSegmentator.cs) 内で OpenCvSharp の `CvDnn.NMSBoxes` を使用して独自のフィルタリング処理を行っています。これにより、UI スライダーからの値が 100% 正確に反映されます。
