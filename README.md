# Diet-ODIN

This repo is the source code of the paper "Diet-ODIN: A Novel Framework for Opioid Misuse Detection with Interpretable Dietary Patterns" Check our paper here, and check our proof-of-concept system demo here (Maintaining, back online soon).

## Environment Settings

 - python==3.8.18
 - pytorch==2.1.0 
 - torch-geometric==2.4.0

To install all requirements for the project using conda:

```
conda env create -f environment.yml
```


## How to run

#### Step 1: Load the data. 

Please download the benchmark graph dataset from this [link](https://drive.google.com/drive/folders/19ZphIEBitMsRjk2A3DLRb5ebpnCzwjbW?usp=sharing), and put the graph under the directory of `processed_data/`. 

This is unnecessary for the reproduction. But the link also contains the structure and supporting files for the graph construction. To download the raw data, please visit the offical NHANES dataset site [here](https://wwwn.cdc.gov/nchs/nhanes/). The data should be put within the structure provided. And you can reproduce the graph executing notebooks under `code/preprocessing/`. 


#### Step 2: Run the experiment.

To reproduce the result, please excute the following command under the `code/` directory. 

```
python main.py 
```

## Citation

## Contact
If you have any questions, don't hesitate to reach out. (zzhang42@nd.edu) 