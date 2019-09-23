"""
   advacned training script
"""
from absl import flags as absl_flags
import tensorflow as tf
import logging
import flags
import time
from tensorflow.python.ops import data_flow_ops

flags.DEFINE_string('network_dir', None, 'network file path.')
flags.DEFINE_string('network', 'network.py', 'network file name')
flags.DEFINE_integer('small_chunk', 1, 'accumulate gradients.')
flags.DEFINE_string('memory_saving_method', None,
                    'setup the memory saving method, 1. recomputing 2. TBD ')
flags.DEFINE_enum('lr_policy', 'multistep', ('multistep', 'exp'), 'learning_rate policy')
flags.DEFINE_boolean('aug_flip', True, 'whether randomly flip left or right dataset')
flags.DEFINE_integer('stop_accu_epoch', 0, 'early stop when accuracy does not increase 1% for'
                     'numbers of epochs')
flags.DEFINE_boolean('save_stop', True, 'whether to save checkpoint when killing process')

from benchmark_cnn import BenchmarkCNN
import benchmark_cnn
from myelindl.core.dataset_lib import AdvancedDataset
from myelindl.core.preprocessing import AdvPreprocessor
from myelindl.core import util
from myelindl.core import const

from myelindl.core import model_lib

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

flags.define_flags()
for name in flags.param_specs.keys():
  absl_flags.declare_key_flag(name)


def maybe_compile(computation, params):
  if params and params.xla_compile:
    return xla.compile(computation)
  else:
    return computation()


class AdvanceCNN(BenchmarkCNN):
    def __init__(self, params):
        dataset = AdvancedDataset(params.data_dir)
        params = params._replace(input_preprocessor='advance',
                                 print_training_accuracy=True)
        if not params.eval and not params.forward_only:
            params = params._replace(eval_during_training_every_n_epochs=1,
                                     num_eval_epochs=1)
        self.model = None
        if params.network_dir is not None:
            self.model = model_lib.AdvModel(params.network_dir, params=params)
        self.local_step = 0
        batch_size = params.batch_size * params.num_gpus
        subset = 'validation' if params.eval else 'train'
        self.num_train_dataset = dataset.num_examples_per_epoch(subset)
        if self.num_train_dataset < batch_size:
            raise ValueError('Batch size larger than total dataset, please reduce batch size.')
        self.num_steps_per_epoch = int(float(self.num_train_dataset+batch_size-1)/batch_size)
        if params.num_epochs:
            num_batches = int(float(params.num_epochs*self.num_train_dataset+batch_size-1) // batch_size)
            params = params._replace(num_epochs=None, num_batches=num_batches)
        if params.num_eval_epochs:
            num_eval_dataset = dataset.num_examples_per_epoch('validation')
            num_eval_batches = int(float(params.num_eval_epochs*num_eval_dataset+batch_size-1)//batch_size)
            params = params._replace(num_eval_epochs=None, num_eval_batches=num_eval_batches)
        display_every = self.num_steps_per_epoch // 10
        if display_every == 0:
            display_every = 1
        if display_every > 100:
            display_every = 100
        if not params.eval:
            params = params._replace(display_every=display_every,
                                     save_summaries_steps=int(self.num_steps_per_epoch))
        super(AdvanceCNN, self).__init__(params, dataset, self.model)
        self.val_accuracy_history = []

    def get_input_preprocessor(self):
        subset = 'validation' if self._doing_eval else 'train'
        preprocessor = AdvPreprocessor(
            self.batch_size * self.batch_group_size,
            self.model.get_input_shapes(subset),
            len(self.devices) * self.batch_group_size,
            dtype=self.model.data_type,
            train=(not self._doing_eval),
            distortions=self.params.distortions,
            resize_method=self.resize_method,
            shift_ratio=0,
            summary_verbosity=self.params.summary_verbosity,
            distort_color_in_yiq=self.params.distort_color_in_yiq,
            fuse_decode_and_crop=self.params.fuse_decode_and_crop)
        ## model.train_input_fn(training_dir, batch_size)
        preprocessor.train_input_fn = self.model.train_input_fn
        ## eval_input_fn(training_dir, batch_size)
        preprocessor.eval_input_fn = self.model.eval_input_fn
        return preprocessor

    def benchmark_one_step(self, sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       partitioned_graph_file_prefix,
                       profiler,
                       image_producer,
                       params,
                       summary_op=None,
                       show_images_per_sec=True,
                       benchmark_logger=None,
                       collective_graph_key=0):
        """Advance one step of benchmarking."""
        run_options = None
        run_metadata = None
        summary_str = None
        start_time = time.time()
        # step before zero accum_grads
        if (self.local_step+1) % self.params.small_chunk == 0:
            if summary_op is None:
                results = sess.run(fetches, options=run_options,
                                   run_metadata=run_metadata)
            else:
                (results, summary_str) = sess.run([fetches, summary_op],
                                                  options=run_options,
                                                  run_metadata=run_metadata)
        else:
            #results = sess.run(self.fetches_forward, options=run_options,
            results = sess.run(fetches, options=run_options,
                               run_metadata=run_metadata)
        if not params.forward_only:
            lossval = results['average_loss']
        else:
            lossval = 0.
        if image_producer is not None:
            image_producer.notify_image_consumption()
        train_time = time.time() - start_time
        step_train_times.append(train_time)
        if (show_images_per_sec and step >= 0 and
            (step == 0 or (step + 1) % params.display_every == 0)):
            speed_mean, speed_uncertainty, speed_jitter = util.get_perf_timing(
                batch_size, step_train_times)
            log_str = '%i\t%s\t%.*f' % (
                step + 1,
                util.get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter),
                const.LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
            logging.info(log_str)
        if benchmark_logger:
            benchmark_logger.log_metric(
                'current_examples_per_sec', speed_mean, global_step=step + 1)
            if 'top_1_accuracy' in results:
                benchmark_logger.log_metric(
                    'top_1_accuracy', results['top_1_accuracy'], global_step=step + 1)
                benchmark_logger.log_metric(
                    'top_5_accuracy', results['top_5_accuracy'], global_step=step + 1)
        self.print_results(results, step)
        self.local_step += 1
        return (summary_str, lossval)

    def print_results(self, results, step):
        if (step >= 0 and (step == 0 or
            (step + 1) % self.params.display_every == 0)):
            epoch = (float(step) * self.batch_size / self.num_train_dataset)
            log_str = 'Training (epoch %.2f):' % (epoch)
            if 'average_loss' in results:
                lossval = results['average_loss']
                log_str += ' loss = %.3f,' % (lossval)
            if 'top_1_accuracy' in results:
                accuracy = results['top_1_accuracy']
                log_str += ' accuracy = %.4f,' % (accuracy)
            if 'learning_rate' in results:
                learning_rate = results['learning_rate']
                log_str += ' learning_rate = %.5f,' % (learning_rate)
            log_str += ' step = {}'.format(step)
            logging.info(log_str)

    def print_eval_results(self, results):
        epoch = float(self.local_step - self.num_warmup_batches)*self.batch_size/self.num_train_dataset
        log_str = 'Validation (epoch %.2f):' % (epoch)
        for key, value in results.iteritems():
            log_str += ' %s = %.4f,' % (key, value)
            if key == 'accuracy':
                self.val_accuracy_history.append(value)
            log_str += ' local_step = {},'.format(self.local_step)
        logging.info(log_str)

    def add_forward_pass_and_gradients(self,
                                     phase_train,
                                     rel_device_num,
                                     abs_device_num,
                                     input_processing_info,
                                     gpu_compute_stage_ops,
                                     gpu_grad_stage_ops):
        """Add ops for forward-pass and gradient computations."""
        nclass = self.dataset.num_classes
        if not self.dataset.use_synthetic_gpu_inputs():
            input_producer_stage = input_processing_info.input_producer_stages[
                rel_device_num]
            with tf.device(self.cpu_device):
                host_input_list = input_producer_stage.get()
            with tf.device(self.raw_devices[rel_device_num]):
                gpu_compute_stage = data_flow_ops.StagingArea(
                    [inp.dtype for inp in host_input_list],
                    shapes=[inp.get_shape() for inp in host_input_list])
                # The CPU-to-GPU copy is triggered here.
                gpu_compute_stage_op = gpu_compute_stage.put(host_input_list)
                input_list = gpu_compute_stage.get()
                gpu_compute_stage_ops.append(gpu_compute_stage_op)
        else:
            with tf.device(self.raw_devices[rel_device_num]):
                # Minor hack to avoid H2D copy when using synthetic data
                input_list = self.model.get_synthetic_inputs(
                    BenchmarkCNN.GPU_CACHED_INPUT_VARIABLE_NAME, nclass)

        # Labels reshaping happens all on gpu:0. Reshaping synthetic labels on
        # multiple devices slows down XLA computation for an unknown reason.
        # TODO(b/116875203): Find/address root cause of XLA slow down.
        labels_device_placement_hack = (
            self.dataset.use_synthetic_gpu_inputs() and self.params.xla_compile)

        def forward_pass_and_gradients():
            build_network_result = self.model.build_network(
                input_list, phase_train, nclass)
            logits = build_network_result.logits

            if not phase_train:
                return [logits]

            base_loss = self.model.loss_function(input_list, build_network_result)
            params = self.variable_mgr.trainable_variables_on_device(
                rel_device_num, abs_device_num)
            l2_loss = None
            total_loss = base_loss
            with tf.name_scope('l2_loss'):
                fp32_params = params
                if self.model.data_type == tf.float16 and self.params.fp16_vars:
                    fp32_params = (tf.cast(p, tf.float32) for p in params)
                if rel_device_num == len(self.devices) - 1:
                    custom_l2_loss = self.model.custom_l2_loss(fp32_params)
                    if custom_l2_loss is not None:
                        l2_loss = custom_l2_loss
                    elif self.params.single_l2_loss_op:
                        reshaped_params = [tf.reshape(p, (-1,)) for p in fp32_params]
                        l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
                    else:
                        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in fp32_params])
            weight_decay = self.params.weight_decay
            if (weight_decay is not None and weight_decay != 0. and
                l2_loss is not None):
                total_loss += len(self.devices) * weight_decay * l2_loss
            aggmeth = tf.AggregationMethod.DEFAULT
            scaled_loss = (total_loss if self.loss_scale is None
                           else total_loss * self.loss_scale)
            grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
            if self.params.sparse_to_dense_grads:
                grads = [
                    grad * tf.cast(1. / self.loss_scale, grad.dtype) for grad in grads
                ]
            ## myelintek ##
            if getattr(self, 'accum_grads', None):
                grads = self.accum_grads(grads, fp32_params)

            if self.params.variable_update == 'horovod':
                import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
                if self.params.horovod_device:
                    horovod_device = '/%s:0' % self.params.horovod_device
                else:
                    horovod_device = ''
                # All-reduce gradients using Horovod.
                grads = [hvd.allreduce(grad, average=False, device_dense=horovod_device)
                         for grad in grads]

            if self.params.loss_type_to_report == 'total_loss':
                loss = total_loss
            else:
                loss = base_loss

            if self.params.print_training_accuracy:
                return [logits, loss] + grads
            else:
                return [loss] + grads

        def unpack_forward_pass_and_gradients_output(forward_pass_and_grad_outputs):
            logits = None
            # logits is only fetched in non-train mode or when
            # print_training_accuracy is set.
            if not phase_train or self.params.print_training_accuracy:
                logits = forward_pass_and_grad_outputs.pop(0)

            loss = (
                forward_pass_and_grad_outputs[0]
                if forward_pass_and_grad_outputs else None)
            grads = (
                forward_pass_and_grad_outputs[1:]
                if forward_pass_and_grad_outputs else None)

            return logits, loss, grads

        def make_results(logits, loss, grads):
            """Generate results based on logits, loss and grads."""
            results = {}  # The return value

            if logits is not None:
                results['logits'] = logits
                accuracy_ops = self.model.accuracy_function(input_list, logits)
                for name, op in accuracy_ops.items():
                    results['accuracy:' + name] = op

            if loss is not None:
                results['loss'] = loss

            if grads is not None:
                param_refs = self.variable_mgr.trainable_variables_on_device(
                    rel_device_num, abs_device_num, writable=True)
                results['gradvars'] = list(zip(grads, param_refs))

            return results

        with tf.device(self.devices[rel_device_num]):
            outputs = maybe_compile(forward_pass_and_gradients, self.params)
            logits, loss, grads = unpack_forward_pass_and_gradients_output(outputs)
            return make_results(logits, loss, grads)
