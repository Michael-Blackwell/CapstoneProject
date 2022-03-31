"""
Author: mike
Created: 3/31/22

"""
from Model import *


ts = pd.Timestamp.now().strftime('%Y-%m-%d_%H.%M.%S')
save_path = Path(f'/media/storage/Capstone1/Models/{ts}')

image_size = (256, 256, 3)
filters = 16
kernel_size = 3
batch_size = 1
recurr = 0

dataset = DataPipe(batch=batch_size, image_size=image_size, output='masks')
dataset.transform_all()

model = build_model(image_size, filters=filters, kernelsize=kernel_size, batch_size=batch_size, recurrence=recurr)

if __name__ == '__main__':
    model = train_model(model, dataset, epochs=10, ts=ts)
    model.save(save_path)
