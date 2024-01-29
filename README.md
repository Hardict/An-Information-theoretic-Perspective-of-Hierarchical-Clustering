# BBM & HCSE
An Information-theoretic Perspective of Hierarchical Clustering related algorithm source code.
## Requirements 
The algorithm is implemented in Python 3.8.
To install the requirements of this project, run the command:

```shell
pip install -r requirements.txt
```
We also use code from other repositoryand made changes.
https://github.com/bgmang/hierarchical-clustering-well-clustered-graphs
- We use its average-linkage, single-linkage, complete-linkage and fix a minor bug.
- We design the cut procedure of BBM algorithm according to its cut algorithm


## Experiments
To evaluate the performance of BBM algorithm,  run the command:
```eval
python BBM.py
```
To evaluate the performance of BBM algorithm, run the command:
```shell
python main.py $cnt_layer $prob $t
```
where $cnt_layer is number of HSBM layers, $prob is a default  probability parameter that you can easily change it in main.py, $t repeats the experiment and number the output file 

A simple batch test script(shell.sh) is given

 