# one-class for LFOC 

## Installation
We use Python 3.6 to code. It needs packages mentioned in `requirements.txt` to run. Please ensure to use torchvision `0.2.1`
```
pip3 install virtualenv
virtualenv myenv
source myenv/bin/activate
pip3 install -r requirements.txt
```

## Data Processing
* Generate close negatives using the [Azure Text to Speech API](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/rest-text-to-speech). (1 sec clips in different accents)
* Download the Audio Commands dataset and generate MFCC features following [this](https://github.com/microsoft/EdgeML/tree/master/examples/pytorch/FastCells/KWS-training). Generate the features for a. the keyword, b. all classes except the keyword, c. close negatives which were generated.
* Use the `code/process_data.py` script to generate the training and testing data. The directory containing the generated files is `root_data` in the following section.

## Example Usage for LFOC Dataset
```
cd code/   
python3 lfoc_main.py  --hd 64 --lr 0.005 --inp_lamda 0.05 --inp_radius 1.5  --batch_size 256 --epochs 150  --seed 0 --one_class_adv 1   --optim 0  --restore 0 -d "root_path"
```

## Arguments Detail
inp_lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)
inp_radius => radius corresponding to the definition of Ni(r)
hd => LSTM Hidden Dimension
one_class_adv => Use the DROCC loss formulation or not (1:Use  0: Dont Use)
optim => 0: Adam   1: SGD(M)

