
# Algorithms

## Examples

- [keras examples](https://github.com/fchollet/keras/tree/master/examples)
- [scikit-learn examples](https://github.com/scikit-learn/scikit-learn/tree/master/examples)
- [tensorflow examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples)

## Tags

- MachineLearning (機械学習)
- NeuralNetwork (ニューラルネットワーク)
- ReinforcedLearning (強化学習)
- Classification (分類)
- Regression (回帰、予測)
- Supervised (教師あり)
- SemiSupervised (半教師あり)
- Unsupervised (教師なし)
- Image
- Text
- Movie
- Value

## Solutions

### Linear Regression

> MachineLearning, Regression, Supervised, Value

- (複数の)特徴数値から出力数値を予測する線形回帰
- [jupyter notebook](https://github.com/arakawamoriyuki/meetup024/blob/master/02_gradient_descent_multiple.ipynb)

### Logistic Regression

> MachineLearning, Classification, Supervised, Value

- (複数の)特徴数値から分類するロジスティック回帰
- [jupyter notebook](https://github.com/arakawamoriyuki/meetup024/blob/master/04_logistic_regression_regularized.ipynb)

### Random Forest

> MachineLearning, Classification, Supervised, Value, Text, Image

- 入力データが一部欠損していても学習可能な分類
- [scikit-learnとgensimでニュース記事を分類する](http://qiita.com/yasunori/items/31a23eb259482e4824e2)
- [jupyter notebook](https://github.com/arakawamoriyuki/meetup024/blob/master/10_random_forest.ipynb)

### K-Means

> MachineLearning, Classification, Unsupervised, Value

- 教師なし分類
- [keras example](https://github.com/scikit-learn/scikit-learn/blob/master/examples/cluster/plot_mini_batch_kmeans.py)
- [jupyter notebook](https://github.com/scikit-learn/scikit-learn/blob/master/examples/cluster/plot_mini_batch_kmeans.py)

### SVM

> MachineLearning?, NeuralNetwork?, Classification, Supervised, Value

- サポートベクタマシン
- 分類

### CNN

> NeuralNetwork, Classification, Supervised, Image

- 画像畳込分類
- [jupyter notebook keras](https://github.com/arakawamoriyuki/meetup024/blob/master/09_mnist_tf_keras.ipynb)
- [jupyter notebook tensorflow](https://github.com/arakawamoriyuki/meetup024/blob/master/07_cnn_with_tensorflow_core.ipynb)
- [TensorFlowでアニメゆるゆりの制作会社を識別する - kivantium活動日記](http://kivantium.hateblo.jp/entry/2015/11/18/233834)
- [畳み込みニューラルネットワークの仕組み | コンピュータサイエンス | POSTD](http://postd.cc/how-do-convolutional-neural-networks-work/)
- [ファッションアイテムの画像からの特徴抽出とマルチスケールなCNNの効果](http://tech.vasily.jp/entry/multiscale_cnn)
- [ファッション×機械学習の論文紹介](http://tech.vasily.jp/entry/fashion_machine_learning_paper)
- [高速な Convolution 処理を目指してみた。　Kn2Image方式](http://qiita.com/t-tkd3a/items/879a5fd6410320fe504e)
- [Convolution処理の手法　Im2Col方式の図解](http://qiita.com/t-tkd3a/items/6b17f296d61d14e12953)
- [tensorflow object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/object_detection/object_detection_tutorial.ipynb)

### RNN

> NeuralNetwork, Regression, Supervised, Text, Image, Movie, Value

- 再帰型データ学習、主に自然言語
- [RNN：時系列データを扱うRecurrent Neural Networksとは](https://deepage.net/deep_learning/2017/05/23/recurrent-neural-networks.html)
- [keras example](https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py)

### LSTM

> NeuralNetwork, Regression, Supervised, Text, Image, Movie, Value

- テキストなど、時系列データ予測
- [再帰型ネットワークと長・短期記憶についての初心者ガイド](https://deeplearning4j.org/ja/lstm)
- [github keras lstm_text_generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)

### CONV LSTM

> NeuralNetwork, Regression, Supervised, Image, Movie

- 画像、動画などの時系列データ分類
- [TensorFlowで畳み込みLSTMを用いた動画のフレーム予測](http://qiita.com/t_shinmura/items/066b696d82f9919480ae)

### Grad-CAM

> NeuralNetwork, Regression, Supervised, Image

- 画像分類ヒートマップ
- [深層学習は画像のどこを見ている！？ CNNで「お好み焼き」と「ピザ」の違いを検証](http://blog.brainpad.co.jp/entry/2017/07/10/163000)

### Deep Photo Style Transfer

> NeuralNetwork, Regression, Supervised, Image

- 画風変換
- [github](https://github.com/luanfujun/deep-photo-styletransfer)
- [paper](https://arxiv.org/abs/1703.07511)

### Tow-Stream CNN

> NeuralNetwork, Classification, Supervised, Movie

- 動画分類

### CNN+LSTM

> NeuralNetwork, Regression, Supervised, Text

- 画像へ説明文を追加

### CNN-SLAM

> NeuralNetwork, Regression, Supervised, Image, Movie

- 画像距離推定 奥行き判定?

### Convolution3DCNN

> NeuralNetwork, Classification, Supervised, Image, Movie

- 三次元データ(連続した画像、動画フレームなど?)畳み込み分類
- [keras reference](https://keras.io/layers/convolutional/#conv3d)
- [3DConvolution + Residual Networkで遊んでみた](http://wazalabo.com/3dconvolution-residual-network.html)

### Semi Supervised?

> MachineLearning, Classification, SemiSupervised, Image

- 一部の答えデータしかない半教師あり学習
- [keras 画像の例](https://github.com/scikit-learn/scikit-learn/blob/master/examples/semi_supervised/plot_label_propagation_digits.py)

### BNN

> NeuralNetwork, Classification, Supervised, Image

- 精度より予測速度を重視する為、データをfloatから0or1のバイナリ化して学習。主に組み込み
- [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ)

### PAF

> MachineLearning?, NeuralNetwork?, Regression, Supervised, Image

- 画像から関節位置の対応付け
- ボーンアニメーション作成に使えそう?
- [openpose github](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### FCN

> NeuralNetwork, Regression, Supervised, Image

- 物体検出、segmentation

### RCNN

> NeuralNetwork, Regression, Supervised, Image

- 物体検出、segmentation
- [Faster R-CNNの紹介](http://kivantium.hateblo.jp/entry/2015/12/25/112145)
- [Recurrent Convolutional NNでテキスト分類](http://qiita.com/knok/items/26224c8489ad681769c0)

### GAN

> NeuralNetwork, Regression, Supervised, Image

- 画像生成系アルゴリズム
- [GAN祭り](http://tech-blog.abeja.asia/entry/everyday_gan)

### DCGAN

> NeuralNetwork, Regression, Supervised, Image

- 画像生成
- [DCGAN-tensorflowで自動画像生成をお手軽に試す](http://qiita.com/shu223/items/b6d8dc1fccb7c0f68b6b)

### iGAN

> NeuralNetwork, Regression, Supervised, Image

- 一部情報を与えて画像生成?
- [github](https://github.com/junyanz/iGAN)

###  WGAN

> NeuralNetwork, Regression, Supervised, Image

- [github](https://github.com/takat0m0/WGAN/blob/master/main.py)

### Interactive Deep Colorization

> NeuralNetwork, Regression, Supervised, Image

- 線画着色
- [github](https://github.com/junyanz/interactive-deep-colorization)

### Light Field Video

> NeuralNetwork, Regression, Supervised, Image, Movie

- 動画や画像の再フォーカス!?
- [github](https://github.com/junyanz/light-field-video)

### Pix2pix

> NeuralNetwork, Regression, Supervised, Image

- 画像から画像を生成
- [github](https://github.com/phillipi/pix2pix)

### Mirror Mirror

> NeuralNetwork, Regression, Supervised, Image

- ビデオからより良い肖像画、証明写真構図を出力する?
- [Mirror Mirror](http://people.eecs.berkeley.edu/~junyanz/projects/mirrormirror/index.html)

### Milcut

> NeuralNetwork, Regression, Supervised?, Image

- オブジェクトを最適な範囲で切り取る
- [Milcut](https://jiajunwu.com/projects/milcut.html)

### CycleGAN

> NeuralNetwork, Regression, Supervised, Image

- 犬を猫にするなど、画像変換
- [画像変換手法CycleGANでパンダ生成モデルを作った話](http://qiita.com/TSY/items/18eb8e9b6342d368c445)

### Globally and Locally Consistent Image Completion

> NeuralNetwork, Regression, Supervised, Image

- 欠損した画像範囲の補完
- [ディープネットワークによるシーンの大域的かつ局所的な整合性を考慮した画像補完](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/ja/)

### DQN

> ReinforcedLearning, Regression, Image, Value

- 強化学習
- [openai/universe](https://github.com/openai/universe)
- [openai/gym](https://github.com/openai/universe)

### Word2vec

> NeuralNetwork, Regression, Supervised, Text

- 自然言語演算
- [Word2Vec：発明した本人も驚く単語ベクトルの驚異的な力](https://deepage.net/bigdata/machine_learning/2016/09/02/word2vec_power_of_word_vector.html)

### Doc2vec

> NeuralNetwork, Regression, Supervised, Text

- 自然言語演算、word2vecの後発

### HMM

> MachineLearning, Regression, Supervised, Text

- Hidden Markov Model
- 自然言語生成
- [マルコフモデル ～概要から原理まで～](http://postd.cc/from-what-is-a-markov-model-to-here-is-how-markov-models-work-1/)
- [scikit-learn hmm](http://scikit-learn.sourceforge.net/stable/modules/hmm.html)

### CF

> MachineLearning, Regression, Supervised, Value

- Collaborative Filtering (コラボレーティブフィルタリング)
- 類似検索、レコメンドアルゴリズム
- [Recommendation System Algorithms](http://www.datasciencecentral.com/profiles/blogs/recommendation-system-algorithms)

### Genetic Algorithm

> Regression, Value

- 遺伝的アルゴリズム(機械学習ではない)
- パラメータをスコアによって収束させる?
- [遺伝的アルゴリズムでナーススケジューリング問題（シフト最適化）を解く](http://qiita.com/shouta-dev/items/1970c2746c3c30f6b39e)
