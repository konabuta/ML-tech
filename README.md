# ML-tech

Microsoft libraries, tools, recipes, ample codes and workshop contents for machine learning & deep learning.


# 1. Library & tool
### [LightGBM](https://github.com/microsoft/LightGBM)

高速勾配ブースティングのライブラリ。Kaggle の成績上位者が良く利用しており、現場でも実績がある。

### [EconML](https://github.com/microsoft/EconML)

統計的因果推論の推定ライブラリ。

### [DoWhy](https://github.com/microsoft/dowhy)
統計的因果推論のライフサイクルをサポートするライブラリ。


### [Nueral Network Intelligence](https://github.com/microsoft/nni)

Neural Architect Search, Hyperparameter Tuning などの AutoML Toolkit。

### [InterpretML](https://github.com/interpretml/interpret)

一般化加法モデルのGA2Mを実装したライブラリ。一般化線形モデルよりも柔軟な設計が可能なため精度向上が期待できる。モデルが解釈可能なことでも注目されている。

### [Interpret Community](https://github.com/interpretml/interpret-community)

様々なモデル解釈のテクノロジーを統合 API 経由で提供。また、専用の Dashboard によりモデルの解釈が可能に。

### [MMLSpark](https://github.com/Azure/mmlspark)

分散コンピューティング環境 Apache Spark 上で動作する機械学習フレームワーク。LightGBM、OpenCV なども利用可能。

### [EdgeML](https://github.com/microsoft/EdgeML)
Edge デバイスのための機械学習のアルゴリズム。

### [Dice](https://github.com/microsoft/DiCE)
反事実 (counterfactual) によるモデルの解釈

### [MMdnn](https://github.com/microsoft/MMdnn)

Deep Neural Network を可視化するクロスフレームワークソリューション

### [TensorWatch](https://github.com/microsoft/tensorwatch)

機械学習のデバック、可視化ツール

### [ONNX Runtime](https://github.com/microsoft/onnxruntime)

ONNX モデルファイルを動作させるランタイム

### [TagAnomaly](https://github.com/microsoft/TagAnomaly)

時系列データ用のタギングツール

### [VoTT](https://github.com/microsoft/VoTT)

画像、動画データ用のタギングツール


### [TextWorld](https://github.com/microsoft/TextWorld)

テキストベースの強化学習のためのゲームシミュレーター


### [AirSim](https://github.com/microsoft/AirSim)

自動運転シミュレーター

# 2. Recipe
### [Computer Vision](https://github.com/microsoft/computervision-recipes)

コンピュータービジョンのベストプラクティス集

### [Neural Language Processing]()

自然言語処理のベストプラクティス集

### [Recommenders](https://github.com/microsoft/recommenders)

推薦システムのベストプラクティス集

### [MLOps](https://github.com/microsoft/MLOps)

MLOps のベストプラクティス集

<br>

# 3. Sample Codes
## Azure Machine Learning 関連

### [Azure ML Sample Codes](https://github.com/microsoft/MachineLearningNotebooks)
Azure Machine Learning 公式サンプルコード。

### [BERT](https://github.com/microsoft/AzureML-BERT)

BERT の E2E の再学習・転移学習のサンプルコード

### [Distributed Deep Learning](https://github.com/microsoft/DistributedDeepLearning)

分散 Deep Learning サンプルコード。

### [Hyperdrive for Deep Learning](https://github.com/microsoft/HyperdriveDeepLearning)

Deep Learning モデルのハイパーパラメータチューニングのサンプルコード。Mask RCNN を利用。

### [Batch Inference](https://github.com/microsoft/Batch-Scoring-Deep-Learning-Models-With-AML)

バッチ推論のサンプルコード。

### [ML on IoT Edge](https://github.com/microsoft/deploy-MLmodels-on-iotedge)

Azure IoT Edge に機械学習モデルをデプロイする手順サンプル


# 4. Workshop

### [Causal Inference and Counterfactual Reasoning @KDD2018](https://causalinference.gitlab.io/kdd-tutorial/)

DoWhy ライブラリのチュートリアル

### [Nvidia Rapids on Azure ML](https://github.com/rapidsai/notebooks-contrib/tree/branch-0.12/conference_notebooks/KDD_2019)

Azure Machine Learning 上で NVidia Rapids を利用するためのチュートリアル。詳細ブログは[こちら](https://medium.com/rapids-ai/rapids-on-microsoft-azure-machine-learning-b51d5d5fde2b)。

### [Deep Learning for Time Series Forecasting @KDD2019](https://github.com/Azure/DeepLearningForTimeSeriesForecasting)

深層学習による時系列予測モデリングのチュートリアル。

### [From Graph to Knowledge Graph @KDD2019](https://github.com/graph-knowledgegraph/KDD2019-HandsOn-Tutorial)

グラフデータとそのモデリングの基礎チュートリアル。

### [AutoML Workshop @Dllab](https://github.com/konabuta/Automated-ML-Workshop)

Azure の AutoML に関するチュートリアル。

### [Deep Learning with TensorFlow 2.0 and Azure @TensorFlow World 2019 ](https://github.com/microsoft/bert-stack-overflow)