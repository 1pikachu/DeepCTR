import numpy as np
import tensorflow as tf

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat,get_feature_names
from deepctr.models import DSIN
import argparse
import time
import os


tf.config.experimental.enable_tensor_float_32_execution(False)

parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="training.")
parser.add_argument("--evaluate", action='store_true', help="evaluation.")
# parser.add_argument("--evaluate", type=0, default='True', help="evaluation.")
parser.add_argument("--predict", action='store_true', help="predict.")
parser.add_argument("--profile", action='store_true', help="profile.")
parser.add_argument("--tensorboard", action='store_true')
parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--precision", type=str, default='float32', help="float32, int8 or float16")
parser.add_argument("--epochs", type=int, default=10, help="training epochs")
parser.add_argument("-i", "-n", "--num_iter", type=int, default=200)
parser.add_argument("--num_warmup", type=int, default=3)
args = parser.parse_args()
print(args)

if args.precision == 'float16' :
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

    from tensorflow.keras import layers
    num_units = 64
    dense1 = layers.Dense(num_units, activation='sigmoid', name='dense_1')
    print(dense1.dtype_policy)

# timeline
import pathlib
timeline_dir = str(pathlib.Path.cwd()) + '/timeline/' + str(os.getpid())


def get_xy_fd(hash_flag=False):
    feature_columns = [SparseFeat('user', 3, embedding_dim=10, use_hash=hash_flag),
                       SparseFeat('gender', 2, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('item', 3 + 1, embedding_dim=4, use_hash=hash_flag),
                       SparseFeat('cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag),
                       DenseFeat('pay_score', 1)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('sess_0_item', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item'),
                         maxlen=4), VarLenSparseFeat(
            SparseFeat('sess_0_cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='cate_id'),
            maxlen=4)]
    feature_columns += [
        VarLenSparseFeat(SparseFeat('sess_1_item', 3 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='item'),
                         maxlen=4), VarLenSparseFeat(
            SparseFeat('sess_1_cate_id', 2 + 1, embedding_dim=4, use_hash=hash_flag, embedding_name='cate_id'),
            maxlen=4)]

    behavior_feature_list = ["item", "cate_id"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    cateid = np.array([1, 2, 2])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    sess1_iid = np.array([[1, 2, 3, 0], [3, 2, 1, 0], [0, 0, 0, 0]])
    sess1_cate_id = np.array([[1, 2, 2, 0], [2, 2, 1, 0], [0, 0, 0, 0]])

    sess2_iid = np.array([[1, 2, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    sess2_cate_id = np.array([[1, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    sess_number = np.array([2, 1, 0])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'cate_id': cateid,
                    'sess_0_item': sess1_iid, 'sess_0_cate_id': sess1_cate_id, 'pay_score': score,
                    'sess_1_item': sess2_iid, 'sess_1_cate_id': sess2_cate_id, }

    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    x["sess_length"] = sess_number
    y = np.array([1, 0, 1])
    return x, y, feature_columns, behavior_feature_list

def pad_dataset(x, y, batch_size):
    total_item = len(y)
    remain_item = total_item % batch_size
    if remain_item > 0:
        pad_item = batch_size - remain_item
        for k in x:
            v = x[k]
            one_item = v[0]
            if len(v.shape) > 1:
                one_item = np.expand_dims(one_item, axis = 0)
            pad_v = np.repeat(one_item, pad_item, axis = 0)
            x[k] = np.concatenate((v, pad_v), axis = 0)
        one_item = y[0]
        if len(y.shape) > 1:
            one_item = np.expand_dims(one_item, axis = 0)
        pad_y = np.repeat(one_item, pad_item, axis = 0)
        y = np.concatenate((y, pad_y), axis = 0)
    return x, y

if __name__ == "__main__":
    if tf.__version__ >= '2.0.0':
        tf.compat.v1.disable_eager_execution()

    x, y, feature_columns, behavior_feature_list = get_xy_fd(True)
    model = DSIN(feature_columns, behavior_feature_list, sess_max_count=2,
                 dnn_hidden_units=[4, 4, 4], dnn_dropout=0.5, )

    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    if args.train:
        print("## Training Start:")
        history = model.fit(x, y, verbose=1, epochs=10, validation_split=0.5)
    if args.evaluate:
        print("## Evaluate Start:")
        total_time = 0.0
        total_sample = 0
        x, y = pad_dataset(x, y, args.batch_size)
        for i in range(args.epochs):
            if args.tensorboard and i == args.epochs // 2:
                print("---- collect tensorboard")
                options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)
                tf.profiler.experimental.start('./tensorboard_data', options = options)
            start_time = time.time()
            model.evaluate(x, y, batch_size=args.batch_size)
            end_time = time.time()
            print("Iteration: {}, inference time: {}".format(i, end_time - start_time), flush=True)
            if i > args.num_warmup:
                total_time += end_time - start_time
                total_sample += len(y)
            if args.tensorboard and i == args.epochs // 2:
                tf.profiler.experimental.stop()
                print("---- collect tensorboard end")
        latency = total_time / total_sample * 1000
        throughput = total_sample / total_time
        print("### Latency:: {:.2f} ms".format(latency))
        print("### inference Throughput: {:.3f} samples/s".format(throughput))

    # if args.predict:
    #     # predict
    #     pred_ans = model.predict(test_model_input, batch_size=256)
    #     print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    #     print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
