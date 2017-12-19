
[PPR Program]

1. InputProgram/<program_name>/<program_name>.ppr
2. InputProgram/<program_name>/<program_name>.cfacts
3. InputProgram/<program_name>/<program_name>.graph
4a. InputProgram/<program_name>/<program_name>-train.examples
4b. InputProgram/<program_name>/<program_name>-test.examples

[ Intermediate Data Processing ]

5. ProcessedData/<program_name>/<program_name>.grounded
6. ProcessedData/<program_name>/Tensors/
7. ProcessedData/<program_name>/<program_name>.weights

[ Code ]

lib/parse.py        [Handles Parsing of PPR program]
lib/unification     [Handles FOL Unification]

build_ground_sld.py [ground sld with features] (Takes 1-4 and outputs 5)
graph_to_tensors.py [Takes 5 and generates annotated Tensors/ for Tensorflow input]
train.py			[Takes Tensors/, learns and dumps weights to sample.weights]

[ PipeLine ]

1. Select a program name and put Program files 1-4 in InputProgram/<program_name>/
2. Set selected program_name in setting.py
2. Run python build_ground_sld.py
3. Run python graph_to_tensors.py
4. train.py
5. evaluate.py 		[ToDo]