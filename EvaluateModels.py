
from ModelFunctions import show_predictions
from OSC.Functions import *
from OSC.Dataset import *
import tensorboard
import pandas as pd
from pathlib import Path
from CustomFunctions import *
from keras.losses import BinaryFocalCrossentropy


if __name__ == '__main__':

    # Set up logging
    ts = pd.Timestamp.now().strftime('%Y-%m-%d_%H.%M.%S')

    # load model Best: Attn_Mask_Focal_2022-04-13_22.20.02
    attn_model = '/media/storage/Capstone1/Models/Attn_Mask_Focal_2022-04-13_22.20.02'
    Unet = '/media/storage/Capstone1/Benchmark_Models/Unet/Callbacks/2022-04-12 16:43:35.361150/2d_unet_decathlon'
    model = tf.keras.models.load_model(attn_model,  # attn_model,  # Unet,
                                       custom_objects={'jaccard_distance': jaccard_distance,
                                                       'dice_coef_loss': dice_coef_loss,
                                                       'myIOU': myIOU,
                                                       'binary_focal_crossentropy': BinaryFocalCrossentropy(),
                                                       'dice_coef': dice_coef,
                                                       "combined_dice_ce_loss": combined_dice_ce_loss,
                                                       "soft_dice_coef": soft_dice_coef,
                                                       })

    # # load and transform dataset
    data_pipe = DataPipe(batch=1, image_size=(512, 512, 3), output='masks')
    test_ds = data_pipe.apply_transformations(data_pipe.test)
    test_ds = test_ds.batch(1)

    # Evaluate Model
    test_results = model.evaluate(test_ds)
    for idx, metric in enumerate(test_results):
        print(f"Test dataset {model.metrics_names[idx]} = {metric}")
    #
    # # Make Predictions
    out_folder = Path.home() / 'Desktop/Test'
    show_predictions(model, dataset=test_ds, num=(1, 15), out_path=out_folder)
