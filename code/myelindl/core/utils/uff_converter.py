#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import uff
from tensorflow.python.tools import freeze_graph


def load_graph(frozen_graph):
    """Load the model from file into GraphDef/Graph object

    Args:
        frozen_graph: The serialization of GraphDef model in the file like *.pbtxt or *.pb

    Returns:
        None
    """

    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # with on prefix name
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def write_model_graph(sess, dest_path='/tmp', output_model_graph_name='model_graph.pbtxt'):
    """Write the model's GraphDef into file

    This model file doesn't contain variables data.

    Args:
        sess: Session object
        dest_path: The destination path of the output UFF file
        output_model_graph_name: The name of output file

    Returns:
        None
    """

    tf.train.write_graph(sess.graph, dest_path, output_model_graph_name)


def do_freeze_graph(input_graph_path, checkpoint_path, output_node_names_str, dest_path='/tmp',
                 output_frozen_graph_name='frozen_graph.pb', input_binary=False):
    """ A wrapper function to use API:freeze_graph.freeze_graph() to generate frozen model file (*.pb).

    This function leverages API:freeze_graph.freeze_graph() and becomes more easy to use.
    The function doesn't conatins any optimization approach inside it.

    Args:
        input_graph_path: The serialization of GraphDef in file, such as *.pbtxt or *.pb
        checkpoint_path: The check-point path that contains variables data
        output_node_names_str: The output operation name list split by comma. 
          for instance: 
              "fasterrcnn/rcnn/rcnn_proposal_1/GatherV2_161,
               fasterrcnn/rcnn/fc_bbox/add,fasterrcnn/rcnn/fc_classifier/add,
               fasterrcnn/rcnn/rcnn_proposal_1/TopKV2"
        dest_path: The destination path of the output UFF file
        output_frozen_graph_name: The name of the frozen graph file
        input_binary: The graph from input_graph_path is binary format or not

    Returns:
        None
    """

    input_saver_def_path = ""
    restore_op_name = ""
    filename_tensor_name = ""
    output_optimized_graph_name = ""
    clear_devices = True

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
        input_binary, checkpoint_path, output_node_names_str,
        restore_op_name, filename_tensor_name,
        os.path.join(dest_path, output_frozen_graph_name),
        clear_devices, "")


def convert_uff_from_tensorflow(sess, graph_def, model_output, dest_path='/tmp', dest_name='converted.uff'):
    """Convert Session GraphDef from TensorFlow to UFF format model
    
    This function is for converting directly from TensorFlow's Session GraphDef and Session object. 
    The session should be built with graph and the variables are also restored from check-point files already.

    Args:
        sess: Session object
        graph: GraphDef object is from either pbtxt file or Python's model source code 
        dest_path: The destination path of the output UFF file
        dest_name: The name of the UFF file

    Returns:
        None
    """
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, model_output)
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)
    #Create UFF model and dump it on disk
    uff_model = uff.from_tensorflow(frozen_graph, model_output)
    dump = open(os.path.join(dest_path, dest_name), 'wb')
    dump.write(uff_model)
    dump.close()


def convert_uff_from_frozen_model(frozen_graph, model_output, dest_path='/tmp', dest_name='converted.uff'):
    """Convert the frozen model file to UFF format model

    This function is for converting directly from frozen model file which is done by freeze_graph(). 
    So this frozen file will contains the serialization of GraphDef and Variables data in const value.

    Args:
        frozen_graph: The frozen model file (*.pb)
        dest_path: The destination path of the output UFF file
        dest_name: The name of the UFF file

    Returns:
        None
    """

    uff_model = uff.from_tensorflow_frozen_model(frozen_graph, model_output)
    dump = open(os.path.join(dest_path, dest_name), 'wb')
    dump.write(uff_model)
    dump.close()


