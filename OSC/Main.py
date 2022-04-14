"""
Author: mike
Created: 3/31/22

"""
from Model import *
from Dataset import DataPipe
import logging
from playsound import playsound




ts = pd.Timestamp.now().strftime('%Y-%m-%d_%H.%M.%S')
save_path = Path(f'/media/storage/Capstone1/Models/Attn_Focal_{ts}')
callback_path = Path(f'/media/storage/Capstone1/Callbacks')
# log_path = Path(f'/media/storage/Capstone1/Logs/{ts}.txt')
#
# logging.basicConfig(level=logging.WARN,
#                     format='%(asctime)s - %(name)s - %(levelname)s | %(message)s',
#                     handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

image_size = (512, 512, 3)
epochs = 30
filters = 16
batch_size = 4

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
model = build_model(image_size, filters=filters)

dataset = DataPipe(batch=batch_size, image_size=image_size, output='masks')
dataset.transform_all()

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    model = train_model(model, dataset, epochs=epochs, ts=ts, callback_path=callback_path)
    model.save(save_path)

    count = 0
    while count < 10:
        count += 1
        playsound('/usr/share/sounds/LinuxMint/stereo/phone-incoming-call.ogg')
