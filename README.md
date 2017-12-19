## ProPPR - python


**Note: This is NOT an official repository for ProPPR.**

This is a python implemention of the following paper. If you find this implemention useful, please cite the below paper.

> *[Programming with Personalized PageRank: A locally groundable first order probabilistic logic](http://arxiv.org/abs/1511.02799)*, William Wang, Kathryn Mazaitis, William Cohen


### Dependencies:
You will need ```python 2.7``` and ```tensorflow``` to run this implementation.

### Instructions to Run


1. Select name of your ProPPR program KB (eg. ``` sample```) and keep it in ```InputProgram/sample/``` directory. You will need the following file:
   + a) ```sample.ppr```
   + b) ```sample.cfacts```
   + c) ```sample.graph```
   + d) ```sample-train.examples```
   + e) ```sample-test.examples```
2. set selected program name in ```setting.py``` to maintain consistency throughout the execution.
3. Run ```python build_ground_sld.py```
3. Run ```python graph_to_tensors.py```
4. Run ```train.py```


### Issues?

Please email me at [hjtrivedi@cs.stonybrook.edu](mailto:hjtrivedi@cs.stonybrook.edu). I will try to fix it as soon as possible.