import os.path
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile
image_path='/Users/jiajianbang1/python/CNN_mutiple_label/images/1.jpg'
output_graph = 'tmp/output_graph.pb'
model_dir = 'model_dir'
model_file_name = 'classify_image_graph_def.pb'
# 创建图
def create_model_graph():
    with tf.Graph().as_default() as graph:
        model_path = os.path.join(model_dir, model_file_name)

        # 读取训练好的Inception-v3模型
        # 谷歌训练好的模型保存在了GraphDef Protocol Buffer中，里面保存了每一个节点取值的计算方法以及变量的取值
        # 加载图
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # 加载读取的Inception-v3模型，并返回数据：输入所对应的张量和计算瓶颈层结果所对应的张量。
            bottleneck_tensor, jpeg_data_tensor = (tf.import_graph_def(
                graph_def, name='',
                return_elements=['pool_3/_reshape:0', 'DecodeJpeg/contents:0']
            ))
    return graph, bottleneck_tensor, jpeg_data_tensor
# 这个函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(session,image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = session.run(bottleneck_tensor, {image_data_tensor: image_data})
    # print(image_data_tensor.name)
    #[1,1,2048]
    bottleneck_values = np.squeeze(bottleneck_values)  #[2048]
    return bottleneck_values
if __name__ == '__main__':
    graph, bottleneck_tensor, jpeg_data_tensor = create_model_graph()   # 把瓶颈层张量，解码后的图片张量，以及inception的图给返回
    with tf.Session(graph=graph) as session:  # 启动会话
        tf.logging.info('Creating bottleneck at ')
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        try:
            bottleneck_values = run_bottleneck_on_image(session,image_data, jpeg_data_tensor, bottleneck_tensor)
        except Exception as e:
            raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                         str(e)))
        print(bottleneck_values)
        bottleneck_values=np.reshape(bottleneck_values,[1,2048] )
        with gfile.FastGFile(output_graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            # 加载读取的Inception-v3模型，并返回数据：输入所对应的张量和计算瓶颈层结果所对应的张量。
            final, jpeg_data_tensor = (tf.import_graph_def(
                graph_def,
                return_elements=['final_result:0', 'input/BottleneckInputPlaceholder:0']
            ))
            print(session.run(final,{jpeg_data_tensor:bottleneck_values}))

