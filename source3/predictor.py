# coding: utf-8
import argparse
import time
import pickle
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from PIL import Image

from model import MLP


model = None
labels = None


def load_model(gpu=-1):
    """
    モデルとラベルデータを読み込む関数
    """
    global model
    global labels

    # モデルをロード
    with open("origin_model.pkl", 'rb') as m:
        model = pickle.load(m)

    # GPUを使う場合の処理
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()

    # ラベルデータの読み込み
    labels = ["Bag", "T-shirt", "Dress"]


def predict_image(image_filename):
    """
    画像を識別する関数
    """
    # --gpuオプションの指定状況を得るための引数パーサーを作成
    parser = argparse.ArgumentParser(description="識別器")
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    args = parser.parse_args()
    
    # モデルの読み込み関数呼び出し
    load_model(gpu=args.gpu)

    # 画像とラベルデータの読み込み
    img = Image.open(image_filename).convert('L')
    img = img.resize((56, 56))
    img = np.asarray(img).astype(np.float32) / 255
    img = img.reshape((1, -1))

    # GPUを使う場合の処理
    if args.gpu >= 0:
        img = chainer.cuda.to_gpu(img)

    # 画像の識別
    pre = model.predictor(chainer.Variable(img))

    # 結果を確率に変換
    pre = F.softmax(pre)

    # 確率トップ5を抽出
    pre_data = zip(pre[0].data, labels)
    top5 = sorted(pre_data, key=lambda x: -x[0])[:5]

    return top5

        
if __name__ == "__main__":
    t1 = time.time()
    top5 = predict_image("test_bag.png")
    t2 = time.time()

    # 結果出力
    for i, data in enumerate(top5):
        print("{0}。名前：{2} / 確率：{1:.5}%".format(i + 1, data[0] * 100, data[1]))

    print("実行時間：{:.3f}秒".format(t2 - t1))
    
