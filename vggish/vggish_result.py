import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from vggish import vggish_input
from vggish import vggish_params
from vggish import vggish_slim

# 设置文件路径
checkpoint_path = '/home/kxy/project/sound-spaces/mmm/vggish/vggish_model.ckpt'  # VGGish模型检查点路径

# 定义全局变量来保存加载的模型
vggish_model = None
sess = None

def load_vggish_model():
    global vggish_model, sess
    if vggish_model is None or sess is None:
        # 创建会话
        sess = tf.Session()

        # 定义 VGGish 模型
        vggish_model = vggish_slim.define_vggish_slim(training=False)

        # 加载检查点
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

    return vggish_model, sess

def extract_vggish_embeddings(path):
    # 加载并处理音频数据，生成 VGGish 的输入格式
    examples_batch = vggish_input.wavfile_to_examples(path)

    # 确保模型已经加载并获取会话
    model, sess = load_vggish_model()

    # 获取输入和输出的 tensor
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

    # 运行模型并生成嵌入
    embeddings = sess.run(embedding_tensor, feed_dict={features_tensor: examples_batch})

    return embeddings
