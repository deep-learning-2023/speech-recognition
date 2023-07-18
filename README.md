# speech-recognition

A project utilising deep learning and ensemble techniques.

The code allows for training of 3-class predictive `(A, B, unknown)` models, based on WAVs containing labeled speech commands.
The 3-class models can then be used to create an n-class ensemble.

## Description of ensembling

Given the classes: `yes`, `no`, `up`, `down` it is possible to create 6 different 3-class models.
Each model is trained on a balanced subset of the dataset, containing the 2 classes it's learning to discriminate (e.g. `yes`, `up`) and a balanced mix of the remaining classes labeled as `unknown`.
After creatig the models they can be combined into an ensemble voting to select the collective inference.

The voting procedure disregards the unknown votes and then takes the class which was most commonly selected by the models.

### Example of voting:

Given the example ensemble inference:
Model: `yes no` - votes yes
Model: `yes up` - votes up
Model: `yes down` - votes yes
Model: `no up` - votes no
Model: `no down` - votes unknown
Model: `up down` - votes unknown

The unkown votes (3) are discarded, and the majority of the ensemble selected `yes` (2), therefore the ensemble inference is selected as `yes`.

## Notes

The ensemble model has improved the classification accuracy in all tested classes in a 10-class speech recognition problem.
The resultant models are much smaller than similarly performing non-ensembled neural networks.
