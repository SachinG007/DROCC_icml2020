# one-class for TimeSeries

## Data Processing
### Audio-Keywords
* Download the Audio Commands dataset and generate MFCC features following [this](https://github.com/microsoft/EdgeML/tree/master/examples/pytorch/FastCells/KWS-training). Generate the features for a. the keyword and b. all classes except the keyword.
* Use the `process_dataset.py` script to generate the training and testing data. The directory containing the generated files is referred to as `root_data` in the following section.

### Epilepsy
* Download the dataset from the UCI Repository [here](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition). This will consists of a `data.csv` file. 
* To generate the training and test data, use the `code/process_dataset_epilepsy.py` script

```
python code/process_dataset_epilepsy.py -d <path to folder with data.csv> -o <output path>
```
The output path is referred to as "root_data" in the following section.


## Example Usage for Epilepsy Dataset
```
python3  main.py --hd 128 --lr 0.00001 --inp_lamda 0.5 --gamma 2 --ascent_step_size 0.1 --inp_radius 10 --batch_size 256 --epochs 200  --optim 0 --restore 0 --metric AUC -d "root_data"
```

## Arguments Detail
inp_lamda => Weightage to the loss from adversarially sampled negative points (\mu in the paper)  
inp_radius => radius corresponding to the definition of Ni(r)  
hd => LSTM Hidden Dimension  
optim => 0: Adam   1: SGD(M)  
ascent_step_size => step size for gradient ascent to generate adversarial anomalies

