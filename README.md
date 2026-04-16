1. README.md
プロジェクトの概要、数理的背景、および使用方法をまとめています。

Markdown
# Infrastructure Maintenance Simulation: Transition Probability & Bayesian Validation

このプロジェクトは、社会基盤施設（トンネル照明等）の維持管理における**予防保全の有効性**を数理統計学的に検証するためのシミュレーション・プラットフォームです。
前田 (2020) の遷移確率モデルをベースに、実務的な観測ノイズや劣化の加速化を考慮した大規模モンテカルロ・シミュレーションを実装しています。

## 概要
本プロジェクトでは、以下の2つの主要な検証を行います。
1. **劣化モデリング**: 微細なタイムステップを用いた連続的な劣化プロセスの再現。
2. **統計的堅牢性の検証**: 観測ノイズが存在する条件下での、頻度主義（カイ二乗検定）とベイズ統計（周辺尤度比）の同一集団捕捉率の比較。

## 主な機能
- **シナリオシミュレーション**: 「等間隔遷移」および「加速劣化」のシナリオ生成。
- **ロバスト統計解析**: サンプルサイズ $N$ の増大に伴う検定手法の感度分析。
- **環境要因の反映**: 固有定数 $C_{km}$ による塩害や材質の違いのシミュレーション。
- **可視化ダッシュボード**: 健全度曲線、遷移劣化度、捕捉率推移図の自動生成。

## 数理的背景
本シミュレーションは、以下のマスター方程式に基づき状態確率の時間発展を計算します。
$$\frac{dP_{m}}{dt}=\sum_{k}(P_{k}\cdot P_{k\rightarrow m})-\sum_{k}(P_{m}\cdot P_{m\rightarrow k})$$

また、大規模データにおける「リンドレーのパラドックス（大標本問題）」を回避するため、ベイズ因子を用いた動的判定境界の妥当性を検証します。

## セットアップ
1. リポジトリをクローン:
   ```bash
   git clone [https://github.com/yourusername/infrastructure-maintenance-sim.git](https://github.com/yourusername/infrastructure-maintenance-sim.git)
   cd infrastructure-maintenance-sim
依存パッケージのインストール:

Bash
pip install -r requirements.txt
使用方法
メインスクリプトを実行すると、シミュレーションが開始され、結果が output/ ディレクトリに保存されます。

Bash
python transition_probability_simulation.py
参考文献
前田 典昭 (2020). 遷移確率の影響度を考慮した予防保全検討について. 山口大学大学院 創成科学研究科.
