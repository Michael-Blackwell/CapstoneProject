"""
Author: mike
Created: 3/31/22

"""
from OSC.Modelv3 import *
from OSC.Dataset import DataPipe
import logging


ts = pd.Timestamp.now().strftime('%Y-%m-%d_%H.%M.%S')
save_path = Path(f'/media/storage/Capstone1/Models/NAdam_Attn_Jac_Post{ts}')
callback_path = Path(f'/media/storage/Capstone1/Callbacks')


image_size = (512, 512, 3)
epochs = 30
filters = 16
batch_size = 16

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
model = compile_model(image_size, filters=filters, model_name=f'NAdam_Attn_Jac_Post{ts}')

dataset = DataPipe(batch=batch_size, image_size=image_size, output='masks')
dataset.transform_all()

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    model = train_model(model, dataset, epochs=epochs, ts=ts, callback_path=callback_path)
    model.save(save_path)
