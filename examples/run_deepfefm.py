import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFEFM
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names

import time
import os
import argparse
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="training.")
parser.add_argument("--evaluate", action='store_true', help="evaluation.")
parser.add_argument("--predict", action='store_true', help="predict.")
parser.add_argument("--profile", action='store_true', help="profile.")
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--precision", type=str, default='float32', help="float32, int8 or float16")
parser.add_argument("--epochs", type=int, default=20, help="training epochs")
parser.add_argument("-i", "-n", "--num_iter", type=int, default=200)
parser.add_argument("--num_warmup", type=int, default=3)
args = parser.parse_args()

if args.precision == 'float16' :
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    from tensorflow.keras import layers
    num_units = 64
    dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
    print(dense1.dtype_policy)


# timeline
import pathlib
timeline_dir = str(pathlib.Path.cwd()) + '/timeline/' + str(os.getpid())

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}
    print("train dataset len: {}, test dataset len: {}".format(
        len(train_model_input), len(test_model_input)))

    # 4.Define Model,train,predict and evaluate
    model = DeepFEFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    if args.train:
        history = model.fit(train_model_input, train[target].values,
                            batch_size=256, epochs=10, verbose=2, validation_split=0.2, )

    if args.evaluate:
        print("## Evaluate Start:")
        total_time = 0.0
        total_sample = 0
        num_iter = int(len(test_model_input) / args.batch_size)
        num_iter = min(num_iter, args.num_iter)
        for i in range(args.epochs):
            if args.profile and i == (args.epochs // 2):
                print("---- collect tensorboard")
                options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)
                tf.profiler.experimental.start('./tensorboard_data', options = options)
            start_time = time.time()
            model.evaluate(test_model_input, steps=num_iter, batch_size=args.batch_size)
            end_time = time.time()
            print("duration: ", end_time - start_time)
            if i > args.num_warmup:
                total_time += end_time - start_time
                total_sample += num_iter * args.batch_size
            if args.profile and i == (args.epochs // 2):
                tf.profiler.experimental.stop()
                print("---- collect tensorboard end")
        latency = total_time / total_sample * 1000
        throughput = total_sample / total_time
        print("### Latency:: {:.2f} ms".format(latency))
        print("### inference Throughput: {:.3f} samples/s".format(throughput))

    if args.predict:
        # predict
        pred_ans = model.predict(test_model_input, batch_size=256)
        print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
        print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
