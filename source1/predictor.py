# coding: utf-8
import argparse
import time
import chainer
from chainer.links import VGG16Layers
from PIL import Image

model = None
labels = None


def load_model(gpu=-1):
    """
    モデルとラベルデータを読み込む関数
    """

    global model
    global labels

    # --gpuオプションの指定状況を得るための引数パーサーを作成
    parser = argparse.ArgumentParser(description="識別器")
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    args = parser.parse_args()

    # モデルをロード(初回実行時にダウンロードが発生)
    model = VGG16Layers()

    print(model.available_layers)
    # GPUを使う場合の処理
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # ラベルデータの読み込み
    label_file = open("synset_words.txt")
    labels = map(lambda x: x[10:], label_file.read().split('\n'))[:-1]


def predict_image(image_filename):
    """
    画像を識別する関数
    """

    load_model()
    # 画像とラベルデータの読み込み
    img = Image.open(image_filename)
    # 画像の識別
    pre = model.predict([img])

    # 確率トップ5を抽出
    pre_data = zip(pre[0].data, labels)
    top5 = sorted(pre_data, key=lambda x: -x[0])[:5]

    return top5

        
if __name__ == "__main__":

    t1 = time.time()
    top5 = predict_image("cat.jpg")
    t2 = time.time()

    # 結果出力
    for i, data in enumerate(top5):
        print("{0}。名前：{2} / 確率：{1:.5}%".format(i + 1, data[0] * 100, data[1]))

    print("実行時間：{:.3f}秒".format(t2 - t1))
    
