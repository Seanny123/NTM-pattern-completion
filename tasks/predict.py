import os
import time
import numpy as np
import tensorflow as tf
from random import randint

from ntm import NTM
from utils import pprint
from ntm_cell import NTMCell

import ipdb

print_interval = 5

def predict(ntm, seq_length, sess, print_=True):
    """ Run the predict task, wherein the network must predict the next input
    given a periodic input, given a trained NTM. It's basically a variant of
    the copy task where the start_symbol and end_symbol depend on the signal"""

    assert seq_length % 2 == 0
    start_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
    start_symbol[0] = 1
    end_symbol = np.zeros([ntm.cell.input_dim], dtype=np.float32)
    end_symbol[1] = 1

    seq = generate_copy_sequence(seq_length, ntm.cell.input_dim - 2)

    feed_dict = {input_:vec for vec, input_ in zip(seq, ntm.inputs)}
    feed_dict.update(
        {true_output:vec for vec, true_output in zip(seq, ntm.true_outputs)}
    )
    feed_dict.update({
        ntm.start_symbol: start_symbol,
        ntm.end_symbol: end_symbol
    })

    input_states = [state['write_w'][0] for state in ntm.input_states[seq_length]]
    output_states = [state['read_w'][0] for state in ntm.get_output_states(seq_length)]

    result = sess.run(ntm.get_outputs(seq_length) + \
                      input_states + output_states + \
                      [ntm.get_loss(seq_length)],
                      feed_dict=feed_dict)

    is_sz = len(input_states)
    os_sz = len(output_states)

    outputs = result[:seq_length]
    read_ws = result[seq_length:seq_length + is_sz]
    write_ws = result[seq_length + is_sz:seq_length + is_sz + os_sz]
    loss = result[-1]

    if print_:
        np.set_printoptions(suppress=True)
        print(" true output : ")
        pprint(seq)
        print(" predicted output :")
        pprint(np.round(outputs))
        print(" Loss : %f" % loss)
        np.set_printoptions(suppress=False)
    else:
        return seq, outputs, read_ws, write_ws, loss

def predict_train(config, sess):
    """Train an NTM for the copy task given a TensorFlow session, which is a
    connection to the C++ backend"""

    if not os.path.isdir(config.checkpoint_dir):
        raise Exception(" [!] Directory %s not found" % config.checkpoint_dir)

    # delimiter flag-like vector inputs indicating the start and end
    # you can see these in the figure examples in the README
    # this is kind of defined redundantly
    start_symbol = np.zeros([config.input_dim], dtype=np.float32)
    start_symbol[0] = 1
    end_symbol = np.zeros([config.input_dim], dtype=np.float32)
    end_symbol[1] = 1

    # initialise the neural turing machine and the neural-net controller thing
    cell = NTMCell(input_dim=config.input_dim,
                   output_dim=config.output_dim,
                   controller_layer_size=config.controller_layer_size,
                   write_head_size=config.write_head_size,
                   read_head_size=config.read_head_size)
    ntm = NTM(cell, sess, config.min_length, config.max_length)

    print(" [*] Initialize all variables")
    tf.initialize_all_variables().run()
    print(" [*] Initialization finished")

    start_time = time.time()
    for idx in xrange(config.epoch):
        # generate a sequence of random length
        seq_length = randint(config.min_length, config.max_length) * 4
        inc_seq, comp_seq = generate_copy_sequence(seq_length, config.input_dim - 2)

        # this somehow associates the desired inputs and outputs with the NTM
        feed_dict = {input_:vec for vec, input_ in zip(inc_seq, ntm.inputs)}
        feed_dict.update(
            {true_output:vec for vec, true_output in zip(comp_seq, ntm.true_outputs)}
        )
        feed_dict.update({
            ntm.start_symbol: start_symbol,
            ntm.end_symbol: end_symbol
        })

        # this runs the session and returns the current training loss and step
        # I'm kind of surprised it returns the step, but whatevs
        _, cost, step = sess.run([ntm.optims[seq_length],
                                  ntm.get_loss(seq_length),
                                  ntm.global_step], feed_dict=feed_dict)

        # how does one use these checkpoints?
        if idx % 100 == 0:
            ntm.save(config.checkpoint_dir, 'copy', step)

        if idx % print_interval == 0:
            print("[%5d] %2d: %.2f (%.1fs)" \
                % (idx, seq_length, cost, time.time() - start_time))

    print("Training Copy task finished")
    return cell, ntm

def generate_predict_sequence(length, bits):
    """make a signal of 0.75 times length as the training data and the complete signal too"""
    assert length % 4 == 0
    inc_seq = np.zeros([length*3/4, bits + 2], dtype=np.float32)
    comp_seq = np.zeros([length*1/4, bits + 2], dtype=np.float32)
    seq = np.zeros([length/2, bits + 2], dtype=np.float32)
    for idx in xrange(length/2):
        # what the hell is with this range?
        seq[idx, 2:bits+2] = np.random.rand(bits).round()

    inc_seq[:length/2, :] = seq
    inc_seq[length/2:length*3/4, :] = seq[:length/4]
    comp_seq = seq[length/4:]
    return (list(inc_seq), list(comp_seq))
