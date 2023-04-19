from audio_data import AudioDataModule, MFCC_transform, wave2vec_transform

data_module = AudioDataModule(data_dir='./', batch_size=1, data_transform=MFCC_transform())
data_module.prepare_data()

lol = data_module.speechcommands_train
print(data_module.get_data_dimensions())
printed = set()
for batch in data_module.train_dataloader():
    elem = batch[1][0].item()
    if elem not in printed:
        print(f'{data_module.get_label_name(elem)} {elem}')
        printed.add(elem)