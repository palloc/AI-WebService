# coding: utf-8
import argparse
import pickle
import numpy as np
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from data_converter import datasetGenerator
from model import MLP


def main():

    # --gpuオプションの指定状況を得るための引数パーサーを作成
    parser = argparse.ArgumentParser(description="識別器作成")
    parser.add_argument("--gpu", "-g", type=int, default=-1)
    args = parser.parse_args()

    # モデルを定義
    model = L.Classifier(MLP(1000, 3))
    
    # GPUを使う場合の処理
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # データセットの読み込み
    dataset_pathes = np.asarray([["./data_set/Bag/", 0], ["./data_set/T-shirt/", 1], ["./data_set/Dress/", 2]])
    train, test = datasetGenerator(dataset_pathes)

    # ディープラーニング用のパラメータ等の設定
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    train_iter = chainer.iterators.SerialIterator(train, 128)
    test_iter = chainer.iterators.SerialIterator(test, 128, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')

    # 学習時に進捗を表示するなど拡張機能の追加
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.ProgressBar())
    
    # 学習開始
    trainer.run()

    # モデルの保存
    model.to_cpu()
    with open("origin_model.pkl", 'wb') as m:
        pickle.dump(model, m)

    
if __name__ == '__main__':
    main()
