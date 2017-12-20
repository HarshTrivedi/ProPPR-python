## ProPPR - python


**Note: This is NOT an official repository for ProPPR.**

This is a python implemention of the following paper. If you find this implemention useful, please cite the below paper.

> *[Programming with Personalized PageRank: A locally groundable first order probabilistic logic](http://arxiv.org/abs/1511.02799)*, William Wang, Kathryn Mazaitis, William Cohen


### Dependencies:
You will need ```python 2.7```, `numpy` and ```tensorflow``` to run this implementation.

### Usage

Select name of your ProPPR program KB (eg. ``` sample```), set it in `setting.py`. Keep the program related files in ```InputProgram/sample/``` directory. You will need the following files in that directory:
   1. sample.ppr
   2. sample.cfacts
   3. sample.graph
   4. sample-train.examples
   5. sample-test.examples

Let's work with Text Categorization program as a working example. 

```
# sample.ppr
predict(Doc,Label) :- isLabel(Label), ab_classify(Doc,Label).
ab_classify(Doc,Label) :- { f(Word,Label): hasWord(Doc,Word) }.
```
**Rule 1:** Predict a particular class label for a particular document if that label is in the list of labels in the database, and if ab\_classify of that document and that label is true.

**Rule 2:** ab_classify (abduce classification) of a particular document and a particular label is always true. When this rule is executed, apply a feature `f` associating each word in the document with that label. This allows the correctness of abduced classification of document `Doc` as `Label` completely determined by similarity of words present in it with label `Label`. Eg. `f(cat, sports)` is likely to have lower weight and `f(football,sports)` should be more. It will be learned automatically based on training examples.

**Facts (1):** As required by Rule 1, we also need facts corresponding to the predicate `isLabel(Label)`.  These are stored in `.cfacts` file. It is a tab seperated file with first column being the predicate name and rest being the arguments. We have ```pos``` and `neg` labels for data. Following is `cfacts` file for our sample program.

```
# sample.cfacts
isLabel	neg	
isLabel	pos	
```

**Facts (2):** As required by Rule 2, we also need facts corresponding to the predicate `hasWord(Doc,Word)`.  These are stored in `.graph` file. File format is same as `.cfacts`. Difference between `.cfacts` and `.graph` is that facts in `.cfacts` are common across all training/testing examples whereas for `.graph` they are training / testing example dependent. Following  are top few facts in our `sample.graph`:

```
# sample.graph
hasWord	train00001	a    
hasWord	train00001	house    
hasWord	train00001	pricy    
hasWord	train00001	doll    
hasWord	train00002	a    
hasWord	train00002	fire    
hasWord	train00002	little    
hasWord	train00002	truck    
hasWord	train00002	red    
...
```

**Examples** We also need positive and negative examples for training. It is a tab seperated file with first column being the query and rest are answers. Answers prefixed with `+` are positive example answers for that query and the ones with `-` are negative examples. We need `sample-train.examples` and `sample-test.examples`. Following are top few lines of training examples file. 

```
# sample-train.examples
predict(train00004,Y)	-predict(train00004,neg)	+predict(train00004,pos)
predict(train00009,Y)	+predict(train00009,neg)	-predict(train00009,pos)
...
```

Now, the input datatset is prepared!

Parsing of the ProPPR program, building dataset of facts and rules and indexing them for quick lookup is handled by `lib/parse.py`. Next, we want to train our model with training examples. However, we need to ground each of the example queries into grounded SLD resolution graphs. We also need to annotate these SLD graphs with dynamically instantiated features from feature templates. All of this, can be done with:

```
python build_ground_sld.py
```

This generates `sample-train.grounded`, `sample-test.grounded` and `sample.features` in `ProcessedData/sample/` directory. For each example in `sample-train.examples`, there is one line in `sample-train.grounded` that encodes the feature annotated sld-resolution graph for that query. Restart edges and self edges are added for Personalize PageRank to work.

For example, following line:
```
predict(train00004,Y)	-predict(train00004,neg)	+predict(train00004,pos)
```
will be converted to:
```
predict(train00004,Y).	1	6	4	6	13	92	1->1:14@1.0	1->2:1@1.0	2->1:14@1.0	2->3:2@1.0	2->5:2@1.0	3->1:14@1.0	3->4:3@1.0,4@1.0,5@1.0,6@1.0,7@1.0	4->1:14@1.0	4->4:8@1.0	5->1:14@1.0	5->6:9@1.0,10@1.0,11@1.0,12@1.0,13@1.0	6->1:14@1.0	6->6:8@1.0
```

Here, Column 1 represents query, column 2 is node_id of query node, column 3 and 4 are node_ids of positive and negative answer solutions. Column 5,6 are number of nodes and edges in SLD graph. Rest of the columns show edge connections and feature annotations: for example, `3->4:3@1.0,4@1.0,5@1.0,6@1.0,7@1.0` say that there is edge between node 3 to node 4 and it has feature ids 3,4,5,6,7 each with intial value 1.0. 

For visualization purpose, the script also generates `sample.features` which enlists all the unique features instantiated during training. For example:

```
# sample.features
...
```


Next, we need to use the feature annotated groundings of the training examples and perform learning using Supervised Personalized PageRank. This has been implemented in Tensorflow. As a proprocessing step, the sld-resolution graphs need to be converted to `numpy` based tensor (matrix) representations of graphs that can be taken as input in tensorflow. This is done with following:

```
python graphs_to_tensors.py
```
It will create a directory `ProcessedData/sample/Tensors/` and store the sparse representations of these graph tensors there.


Lastly, the learning is done using following script:

```
python train.py
```

As the script runs, it logs the dropping loss value and increasing MMR/AUC with increase in epoch. For our sample dataset, only 1 epoch is more than enough for reaching perfect AUC and MMR. Following is the expected output.

```
<.. fill this up ...>
```

Once 10 epochs are complete, it automatically saves the best performing model in `ProcessedData/sample/model/`. This trained model can be used later anytime.

The script `train.py` can be run in 2 modes: `Learn` or `Predict`. Default is `Learn`  mode which we just used to create the model. `Predict` mode can be used check AUC/MMR on test or train data. The modes can be toggled with variable `run` in first few lines of the script.



### Issues?

Please email me at [hjtrivedi@cs.stonybrook.edu](mailto:hjtrivedi@cs.stonybrook.edu). I will try to fix it as soon as possible.
