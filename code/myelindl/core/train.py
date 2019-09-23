import click
import json
import os
import sys
import tensorflow as tf
import time

from tensorflow.python import debug as tf_debug

from luminoth.train import run
#from luminoth.datasets import get_dataset
from luminoth.datasets.exceptions import InvalidDataDirectory
from luminoth.models import get_model
from luminoth.utils.config import get_config
from luminoth.utils.hooks import ImageVisHook, VarVisHook
from luminoth.utils.training import get_optimizer, clip_gradients_by_norm
from luminoth.utils.experiments import save_run

# benchmarks
import batch_allreduce
from myelindl.core.dataset import get_dataset



def run_local(config, environment=None):
    model_class = get_model(config.model.type)
    image_vis = config.train.get('image_vis')
    var_vis = config.train.get('var_vis')

    if config.train.get('seed') is not None:
        tf.set_random_seed(config.train.seed)

    if config.train.debug or config.train.tf_debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)

    model = model_class(config)

    num_gpus = config.train.get('num_gpus')
    if num_gpus is None:
        num_gpus = 1
    gpu_devices = ['gpu:{}'.format(i) for i in range(num_gpus)]
    gpu_indices = [i for i in range(num_gpus)]

    global_step = tf.train.get_or_create_global_step()

    optimizer = get_optimizer(config.train, global_step)

    def forward_pass_and_gradients(train_dataset):
        """
        Create forward loss and grads on each device
        """
        train_image = train_dataset['image']
        train_filename = train_dataset['filename']
        train_bboxes = train_dataset['bboxes']

        prediction_dict = model(train_image, train_bboxes, is_training=True)
        total_loss = model.loss(prediction_dict)

        # TODO: Is this necesarry? Couldn't we just get them from the
        # trainable vars collection? We should probably improve our
        # usage of collections.
        trainable_vars = model.get_trainable_vars()

        # Compute, clip and apply gradients
        with tf.name_scope('gradients'):
            grads_and_vars = optimizer.compute_gradients(
                total_loss, trainable_vars
            )

            if config.train.clip_by_norm:
                grads_and_vars = clip_gradients_by_norm(grads_and_vars)

        return prediction_dict, total_loss, grads_and_vars


    def build_train_ops(device_grads):
        training_ops = []
        # average all gradients
        grads_to_reduce = [[g for g, _ in grad_vars] for grad_vars in device_grads]
        algorithm = batch_allreduce.AllReduceSpecAlgorithm('nccl', gpu_indices, 0, 10)
        reduced_grads, _ = algorithm.batch_all_reduce(grads_to_reduce, 0, 0, 0)
        reduced_device_grads = [[
            (g, v) for g, (_, v) in zip(grads, grad_vars)
        ] for grads, grad_vars in zip(reduced_grads, device_grads)]

        for i, device in enumerate(gpu_devices):
            with tf.device(device):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(
                        reduced_device_grads[i], global_step=global_step
                    )
                    training_ops.append(train_op)
        train_ops = tf.group(*(training_ops), name='train_ops_group')
        return train_ops


    try:
        dataset_class = get_dataset(config.dataset.type)
        dataset = dataset_class(config)
    except InvalidDataDirectory as exc:
        tf.logging.error(
            "Error while reading dataset, {}".format(exc)
        )
        sys.exit(1)
    device_losses = []
    device_gradvars = []

    for device in gpu_devices:
        train_dataset = dataset()
        with tf.device(device):
            prediction_dict, loss, gradvars = forward_pass_and_gradients(train_dataset)
            device_losses.append(loss)
            device_gradvars.append(gradvars)

    train_filename = train_dataset['filename']

    train_op = build_train_ops(device_gradvars)
    # average losses
    average_loss = tf.reduce_mean(device_losses)


    # Create custom init for slots in optimizer, as we don't save them to
    # our checkpoints. An example of slots in an optimizer are the Momentum
    # variables in MomentumOptimizer. We do this because slot variables can
    # effectively duplicate the size of your checkpoint!
    trainable_vars = model.get_trainable_vars()
    slot_variables = [
        optimizer.get_slot(var, name)
        for name in optimizer.get_slot_names()
        for var in trainable_vars
    ]
    slot_init = tf.variables_initializer(
        slot_variables,
        name='optimizer_slots_initializer'
    )

    # Create saver for saving/restoring model
    model_saver = tf.train.Saver(
        set(tf.global_variables()) - set(slot_variables),
        name='model_saver',
        max_to_keep=config.train.get('checkpoints_max_keep', 1),
    )

    # Create saver for loading pretrained checkpoint into base network
    base_checkpoint_vars = model.get_base_network_checkpoint_vars()
    checkpoint_file = model.get_checkpoint_file()
    if base_checkpoint_vars and checkpoint_file:
        base_net_checkpoint_saver = tf.train.Saver(
            base_checkpoint_vars,
            name='base_net_checkpoint_saver'
        )

        # We'll send this fn to Scaffold init_fn
        def load_base_net_checkpoint(_, session):
            base_net_checkpoint_saver.restore(
                session, checkpoint_file
            )
    else:
        load_base_net_checkpoint = None


    tf.logging.info('Starting training for {}'.format(model))

    run_options = None
    if config.train.full_trace:
        run_options = tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE
        )

    # Create custom Scaffold to make sure we run our own init_op when model
    # is not restored from checkpoint.
    summary_op = [model.summary]
    summaries = tf.summary.merge_all()
    if summaries is not None:
        summary_op.append(summaries)
    summary_op = tf.summary.merge(summary_op)

    # `ready_for_local_init_op` is hardcoded to 'ready' as local init doesn't
    # depend on global init and `local_init_op` only runs when it is set as
    # 'ready' (an empty string tensor sets it as ready).
    is_chief = True
    local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    variable_mgr_init_ops.extend([table_init_ops])
    variable_mgr_init_ops.extend([slot_init])
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)
    scaffold = tf.train.Scaffold(
        saver=model_saver,
        init_op=tf.global_variables_initializer() if is_chief else tf.no_op(),
        local_init_op=local_var_init_op_group,
        ready_for_local_init_op=tf.constant([], dtype=tf.string),
        summary_op=summary_op,
        init_fn=load_base_net_checkpoint,
    )

    # Custom hooks for our session
    hooks = []
    chief_only_hooks = []

    if config.train.tf_debug:
        debug_hook = tf_debug.LocalCLIDebugHook()
        debug_hook.add_tensor_filter(
            'has_inf_or_nan', tf_debug.has_inf_or_nan
        )
        hooks.extend([debug_hook])

    if not config.train.job_dir:
        tf.logging.warning(
            '`job_dir` is not defined. Checkpoints and logs will not be saved.'
        )
        checkpoint_dir = None
    elif config.train.run_name:
        # Use run_name when available
        checkpoint_dir = os.path.join(
            config.train.job_dir, config.train.run_name
        )
    else:
        checkpoint_dir = config.train.job_dir

    should_add_hooks = (
        config.train.display_every_steps
        or config.train.display_every_secs
        and checkpoint_dir is not None
    )
    if should_add_hooks:
        if not config.train.debug and image_vis == 'debug':
            tf.logging.warning('ImageVisHook will not run without debug mode.')
        elif image_vis is not None:
            # ImageVis only runs on the chief.
            chief_only_hooks.append(
                ImageVisHook(
                    prediction_dict,
                    image=train_dataset['image'],
                    gt_bboxes=train_dataset['bboxes'],
                    config=config.model,
                    output_dir=checkpoint_dir,
                    every_n_steps=config.train.display_every_steps,
                    every_n_secs=config.train.display_every_secs,
                    image_visualization_mode=image_vis
                )
            )

        if var_vis is not None:
            # VarVis only runs on the chief.
            chief_only_hooks.append(
                VarVisHook(
                    every_n_steps=config.train.display_every_steps,
                    every_n_secs=config.train.display_every_secs,
                    mode=var_vis,
                    output_dir=checkpoint_dir,
                    vars_summary=model.vars_summary,
                )
            )

    step = -1
    target=''
    config_proto = tf.ConfigProto()
    config_proto.allow_soft_placement = True
    with tf.train.MonitoredTrainingSession(
        master=target,
        is_chief=is_chief,
        checkpoint_dir=checkpoint_dir,
        scaffold=scaffold,
        hooks=hooks,
        chief_only_hooks=chief_only_hooks,
        save_checkpoint_secs=config.train.save_checkpoint_secs,
        save_summaries_steps=config.train.save_summaries_steps,
        save_summaries_secs=config.train.save_summaries_secs,
        config=config_proto,
    ) as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                before = time.time()
                _, train_loss, step, filename = sess.run([
                    train_op, average_loss, global_step, train_filename
                ], options=run_options)

                # TODO: Add image summary every once in a while.

                tf.logging.info(
                    'step: {}, file: {}, train_loss: {}, in {:.2f}s'.format(
                        step, filename, train_loss,
                        time.time() - before
                    ))

                if is_chief and step == 1:
                    # We save the run after first batch to make sure everything
                    # works properly.
                    save_run(config, environment=environment)

        except tf.errors.OutOfRangeError:
            tf.logging.info(
                '{}finished training after {} epoch limit'.format(
                    log_prefix, config.train.num_epochs
                )
            )

            # TODO: Print summary
        finally:
            coord.request_stop()

        # Wait for all threads to stop.
        coord.join(threads)

        return step


@click.command(help='Train models')
@click.option('config_files', '--config', '-c', required=True, multiple=True, help='Config to use.')  # noqa
@click.option('--job-dir', help='Job directory.')
@click.option('override_params', '--override', '-o', multiple=True, help='Override model config params.')  # noqa
def train(config_files, job_dir, override_params):
    """
    Parse TF_CONFIG to cluster_spec and call run() function
    """
    # TF_CONFIG environment variable is available when running using gcloud
    # either locally or on cloud. It has all the information required to create
    # a ClusterSpec which is important for running distributed code.
    tf_config_val = os.environ.get('TF_CONFIG')

    if tf_config_val:
        tf_config = json.loads(tf_config_val)
    else:
        tf_config = {}

    cluster = tf_config.get('cluster')
    job_name = tf_config.get('task', {}).get('type')
    task_index = tf_config.get('task', {}).get('index')
    environment = tf_config.get('environment', 'local')

    # Get the user config and the model type from it.
    try:
        config = get_config(config_files, override_params=override_params)
    except KeyError:
        # Without mode type defined we can't use the default config settings.
        raise KeyError('model.type should be set on the custom config.')

    if job_dir:
        override_params += ('train.job_dir={}'.format(job_dir), )

    # If cluster information is empty or TF_CONFIG is not available, run local
    if job_name is None or task_index is None:
        return run_local(
            config, environment=environment
        )
