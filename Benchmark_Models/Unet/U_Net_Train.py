#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

"""
This module loads the data from data.py, creates a TensorFlow/Keras model
from model.py, trains the model on the data, and then saves the
best model.

From Intel U-Net, modified to run on ISIC 2017 dataset:
https://github.com/IntelAI/unet
"""

import datetime
import os
import tensorboard
from TFModelFunctions import DataPipe
from pathlib import Path
import tensorflow as tf  # conda install -c anaconda tensorflow
import unet.D2.settings as settings  # Use the custom settings.py file for default parameters
import numpy as np
from unet.D2.argparser import args
# TODO use jaccard  & focal loss
# TODO view output masks

"""
For best CPU speed set the number of intra and inter threads
to take advantage of multi-core systems.
See https://github.com/intel/mkl-dnn
"""
ts = datetime.datetime.now()
callback_path = Path(f'/media/storage/Capstone1/Benchmark_Models/Unet/Callbacks/{ts}')


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Get rid of the AVX, SSE warnings

# If hyperthreading is enabled, then use
os.environ["KMP_AFFINITY"] = "granularity=thread,compact,1,0"

# If hyperthreading is NOT enabled, then use
# os.environ["KMP_AFFINITY"] = "granularity=thread,compact"

os.environ["KMP_BLOCKTIME"] = str(args.blocktime)

os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
os.environ["INTRA_THREADS"] = str(args.num_threads)
os.environ["INTER_THREADS"] = str(args.num_inter_threads)
os.environ["KMP_SETTINGS"] = "0"  # Show the settings at runtime


if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    print("Started script on {}".format(START_TIME))

    print("Runtime arguments = {}".format(args))
    # test_intel_tensorflow()  # Print if we are using Intel-optimized TensorFlow

    """
    Create a model, load the data, and train it.
    """

    """
    Step 1: Define a data loader
    """
    print("-" * 30)
    print("Loading the data from the Medical Decathlon directory to a TensorFlow data loader ...")
    print("-" * 30)

    # Build dataset
    ds = DataPipe(batch=1, output='masks')
    ds.transform_all()


    print("-" * 30)
    print("Creating and compiling model ...")
    print("-" * 30)

    """
    Step 2: Define the model
    """
    from unet.D2.model import unet

    unet_model = unet(channels_first=args.channels_first,
                      fms=args.featuremaps,
                      output_path=args.output_path,
                      inference_filename=args.inference_filename,
                      learning_rate=args.learningrate,
                      weight_dice_loss=args.weight_dice_loss,
                      use_upsampling=args.use_upsampling,
                      use_dropout=args.use_dropout,
                      print_model=args.print_model)

    model = unet_model.create_model(
        ds.image_size, ds.image_size)

    model_filename, model_callbacks = unet_model.get_callbacks(callback_path)

    """
    Step 3: Train the model on the data
    """
    print("-" * 30)
    print("Fitting model with training data ...")
    print("-" * 30)

    # Launch Tensorboard, can be accessed by going to http://localhost:6006 in your browser
    tb = tensorboard.program.TensorBoard()
    tb_path = callback_path
    tb.configure(argv=[None, '--logdir', str(tb_path)])
    url = tb.launch()

    model.fit(ds.train,
              epochs=args.epochs,
              validation_data=ds.val,
              verbose=1,
              callbacks=model_callbacks)

    """
    Step 4: Evaluate the best model
    """
    print("-" * 30)
    print("Loading the best trained model ...")
    print("-" * 30)

    unet_model.evaluate_model(model_filename, ds.test)
