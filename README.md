Here we provide the code on the BRACS dataset. It contains detailed implementations of DECOMP as well as all comparison methods. Code on other evaluated datasets (Cityscapes and KiTS23) follow the same implementations.


## Installation
Instructions on env installation, as well as possible bugs with solutions, can be found in [install.txt](install.txt).

## Usage
```python
python experiments.py --exp-id=0 --image-sampling-strategy="decomposition_threshold_0.7" --region-sampling-strategy="decomposition_threshold_0.7" --n-query=1 --max-query-per-WSI=15 
```