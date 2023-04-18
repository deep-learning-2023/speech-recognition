from audio_data import AudioDataModule, MFCC_transform, wave2vec_transform

data_module = AudioDataModule(data_dir='./', batch_size=64, data_transform=wave2vec_transform())
data_module.prepare_data()

print(data_module.get_data_dimensions())

for batch in data_module.train_dataloader():
    print(batch[0].shape)
    print(batch[1].shape)