
import tensorflow as tf

import os
from tensorflow.python import ops


# Source https://medium.com/google-cloud/optimizing-tensorflow-models-for-serving-959080e9ddbf
# Convert the frozen graph to TF model

# Note -You need to change the outputs for your model
# Helper for Converting Frozen graph from Disk to TF serving compatible Model
def get_graph_def_from_file(graph_filepath):
    ops.reset_default_graph()
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def


def convert_graph_def_to_saved_model(export_dir, graph_filepath):
    graph_def = get_graph_def_from_file(graph_filepath)
    sess = tf.Session(graph=ops.Graph())
    with sess as session:
        tf.import_graph_def(graph_def, name='')
        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={'input_image': session.graph.get_tensor_by_name('{}:0'.format(node.name))
                    for node in graph_def.node if node.op == 'Placeholder'},
            outputs={t: session.graph.get_tensor_by_name(t) for t in outputs}

        )
        print('Optimized graph converted to SavedModel!')

# Helper For getting the model Size
import os
from tensorflow.python import ops
def get_size(model_dir, model_file='saved_model.pb'):
  model_file_path = os.path.join(model_dir, model_file)
  print(model_file_path, '')
  pb_size = os.path.getsize(model_file_path)
  variables_size = 0
  if os.path.exists(
      os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
    variables_size = os.path.getsize(os.path.join(
        model_dir,'variables/variables.data-00000-of-00001'))
    variables_size += os.path.getsize(os.path.join(
        model_dir,'variables/variables.index'))
  print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
  print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
  print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))


# You need to change the outputs for your model
outputs = ['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',\
'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',     \
'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0']

gdef = get_graph_def_from_file(r'D:\UNIMA\Retina_FUNCIONAL\snapshots\freeze_model\new_card_v2.pb')
convert_graph_def_to_saved_model('TFS_test_model_2', r'D:\UNIMA\Retina_FUNCIONAL\snapshots\freeze_model\new_card_v2.pb')