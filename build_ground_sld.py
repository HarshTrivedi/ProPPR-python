from copy import deepcopy
from lib.parse import *
import lib.unification as unification
from setting import *
import networkx as nx
import pickle
# Are features specific to 1 sld graph??
# No I guess they are shared variables among all the SLD graphs

# Note: Input of Sld would be a line like the following:
# predict(train00004,Y)	-predict(train00004,neg)	+predict(train00004,pos)

# we need to ground graphs using Sld for each of the (11) examples in sample.examples (for training purpose)
# Given one query example as follows:
# predict(train00004,Y)	-predict(train00004,neg)	+predict(train00004,pos)
# you need to construct query:   predict(train00004,Y)
# and mark those solutions nodes -ve which results in Y=neg
# and mark those solution nodes  +ve which results in Y=pos

# feature id's are global for all examples in sample.examples
# note that we will have 11 groundings from sample.examples.


# INPUT
# input of this

class Sld:
	feature2idx = {'unk': 1} # have to be static attribute
	idx2feature = {1: 'unk'} # have to be static attribute

	# query: predict(train00004,Y) in above code
	# true = {Y: neg} and false = {Y: pos} in above case.
	# note relevant parameter of query and true/false dicts should be tied together.
	def __init__(self, query, db, positives, negatives, mode):
		self.db    = db
		self.query = query

		digraph = nx.DiGraph()

		root_node_id = len(digraph.nodes())+1
		digraph.add_node(root_node_id)
		
		nx.set_node_attributes( digraph, 
								'goals', 
								{root_node_id: [query]}  )
		# nodes should contain goal: list of subgoals
		nx.set_node_attributes( digraph, 
								'substitution',
								{ root_node_id: {} } )

		self.graph = digraph
		self.stack = [root_node_id]
		self.root_node_id = root_node_id
		# append to push
		# pop to pop

		# self.ground_features = set()
		
		self.positives  = positives
		self.negatives  = negatives

		self.pos_nodes = set()
		self.neg_nodes = set()
		self.mode = mode


	def step(self):

		if not self.stack:
			return False

		node_id = self.stack[-1]# .pop()
		# node = self.graph.node(node_id)
		node_goals = nx.get_node_attributes(self.graph,'goals')[node_id]

		# 
		if node_goals:

			subgoal = node_goals[0]

			# use subgoal and 
			# select any rule (rule/fact) that is unifiable
			# unify subgoal with it. Apply same substitution on 
			# rest of the subgoals like node_goals[1:]			
			# and then add a new_node with new subgoals
			# note that in case of rule, new subgoals might be added as well
			# ie. unifying with head and propogating the unifcation to body terms

			tokens = [subgoal.relation.name] #+ [ x.name for x in subgoal.arguments if x.name[0].islower() ]
			rel_db_rules = db.get_rules_with_tokens(tokens)

			for rule_idx, rule in enumerate(rel_db_rules):

				rule = Database.rename_rule(rule)
				head, body, feature = rule
				subgoal_copy = deepcopy(subgoal)
				head_copy = deepcopy(head)
				body_copy = deepcopy(body)

				mgu = unification.unify( subgoal_copy, head_copy )
				if mgu is not None:

					break_substitution = False

					for edge in self.graph.out_edges(node_id, data = True):
						if edge[2]['mgu'] == mgu:
							break_substitution = True
							break
					if break_substitution:
						continue

					subgoals_added = unification.apply_substitution( body, mgu )
					rest_subgoals = deepcopy(node_goals[1:])
					rest_subgoals = unification.apply_substitution(rest_subgoals, mgu)
					if len(feature) == 0:

						# it's a fact
						if len(body) == 0:
							feat_name = 'fact'
						else:
							feat_name = 'rule_{}'.format(rule_idx)

						if self.mode == 'train':
							if feat_name in Sld.feature2idx:
								idx = Sld.feature2idx[feat_name]
							else:
								idx = len(Sld.feature2idx)+1
								Sld.feature2idx[feat_name] = idx
								Sld.idx2feature[idx] = feat_name
						else:
							feat_name = 'unk'
							idx = Sld.feature2idx[feat_name]

						ground_features = [feat_name]
					elif len(feature) == 1:
						# print 'In 1'

						ground_features = []
						feature = unification.apply_substitution(feature, mgu)

						tokens = [feature[0].relation.name] + [ x.name for x in feature[0].arguments if x.name[0].islower() ]
						rel_rules = db.get_rules_with_tokens(tokens)
						for rule_f in rel_rules:
							if rule_f[1] == [] and rule_f[2] == []:
								feature_copy = deepcopy(feature)
								subs = unification.unify(rule_f[0],feature[0])
								if subs is not None:
									ground_feature = unification.apply_substitution( feature, subs )[0]
									ground_features.append(ground_feature)
					else:
						# print 'In 2'
						ground_features = []
						feature = unification.apply_substitution(feature, mgu)

						tokens = [feature[1].relation.name] + [ x.name for x in feature[1].arguments if x.name[0].islower() ]
						rel_rules = db.get_rules_with_tokens(tokens)
						for rule_f in rel_rules:
							if rule_f[1] == [] and rule_f[2] == []:
								feature_copy = deepcopy(feature)
								subs = unification.unify( rule_f[0], feature_copy[1] )
								if subs is not None:
									ground_feature = unification.apply_substitution( feature_copy, subs )[0]
									ground_features.append(ground_feature)

					feature_ids = []
					for ground_feature in ground_features:
						# make a set and id for ground features (should be static)

						if self.mode == 'train':
							if str(ground_feature) in Sld.feature2idx:
								idx = Sld.feature2idx[ str(ground_feature) ]
							else:
								idx = len(Sld.feature2idx)+1
								Sld.feature2idx[str(ground_feature)] = idx
								Sld.idx2feature[idx] = str(ground_feature)
						else:
							feat_name = 'unk'
							idx = Sld.feature2idx[feat_name]
						feature_ids.append(idx)

					new_node_goals = subgoals_added + rest_subgoals
					new_node_id = len(self.graph.nodes()) + 1

					self.graph.add_node(new_node_id)
					nx.set_node_attributes( self.graph, 
											'goals',
											{ new_node_id: deepcopy(new_node_goals) } )

					self.graph.add_edge(node_id, 
										new_node_id, 
										mgu = mgu, 
										feat = feature_ids)

					previous_subst = nx.get_node_attributes(self.graph, 'substitution')[node_id]
					new_node_subst = unification.compose_mgus( previous_subst, mgu )
					nx.set_node_attributes( self.graph, 
											'substitution',
											{ new_node_id: new_node_subst } )


					if not new_node_goals:

						for positive in self.positives:
							is_positive = all((k in new_node_subst and new_node_subst[k]==v) for k,v in positive.iteritems())
							if is_positive:
								nx.set_node_attributes( self.graph, 
													'label',
													{ new_node_id: 1 } )
								self.pos_nodes.add(new_node_id)

						for negative in self.negatives:
							is_negative = all((k in new_node_subst and new_node_subst[k]==v) for k,v in negative.iteritems())
							if is_negative:
								nx.set_node_attributes( self.graph, 
													'label',
													{ new_node_id: -1 } )
								self.neg_nodes.add(new_node_id)

						if self.mode == 'train':
							feat_name = 'solution_return'
							if feat_name in Sld.feature2idx:
								idx = Sld.feature2idx[feat_name]
							else:
								idx = len(Sld.feature2idx)+1
								Sld.feature2idx[feat_name] = idx
								Sld.idx2feature[idx] = feat_name
						else:
							feat_name = 'unk'
							idx = Sld.feature2idx[feat_name]

						self.graph.add_edge(new_node_id, 
								new_node_id, 
								mgu = {}, 
								feat = [idx])

					self.stack.append(new_node_id)

					return True 
			else:
				self.stack.pop()
				self.step()
				return True 

		else:
			self.stack.pop()
			self.step()
			return True 


	def add_back_edges(self):
		alpha = 0.1
		query_node = 1
		for node_id in self.graph.nodes():
			degree = self.graph.out_degree(node_id)
			weight = (degree * alpha / (1-alpha))

			if self.mode == 'train':
				feat_name = 'restart'
				if feat_name in Sld.feature2idx:
					idx = Sld.feature2idx[feat_name]
				else:
					idx = len(Sld.feature2idx)+1
					Sld.feature2idx[feat_name] = idx
					Sld.idx2feature[idx] = feat_name
			else:
				feat_name = 'unk'
				idx = Sld.feature2idx[feat_name]

			self.graph.add_edge(node_id, 
					query_node, 
					mgu = {}, 
					feat = [idx] )
		return None


	def dump(self):

		query = str(self.graph.node[1]['goals'][0]) + '.'

		query_node = str(1)
		pos_nodes = ','.join(map(str,self.pos_nodes))
		neg_nodes = ','.join(map(str,self.neg_nodes))

		nodes_count = str(len( self.graph.nodes() ))
		edges_count = str(len( self.graph.edges() ))

		features_count = len(self.idx2feature.keys())
		label_dependency_count = str(features_count)

		edge_marks = []
		for edge in self.graph.edges(data = True):
			feats = edge[2]['feat']
			source, target = str(edge[0]), str(edge[1])
			edge_mark = '->'.join([source, target])
			edge_mark = edge_mark + ':' + ','.join([ '@'.join([str(x),str(y)]) for x,y  in zip(feats, ['1.0']*len(feats))])
			edge_marks.append(edge_mark)

		edge_marks_str = '\t'.join(edge_marks)
		line = '\t'.join([query, query_node, pos_nodes, neg_nodes, nodes_count, edges_count, label_dependency_count, edge_marks_str ])

		return line


if __name__ == "__main__":

	input_dir = './InputProgram'

	db = Database(program_name, input_dir)
	db.load_ppr()
	db.load_cfacts()
	db.load_graph()
	db.build_index()	

	processed_data_dir = os.path.join('ProcessedData', program_name)
	if not os.path.exists(processed_data_dir):
		os.makedirs(processed_data_dir)

	set_names = ['train', 'test']
	for set_name in set_names:
		print 'In set {}'.format(set_name)
		examples_path = db.path_train_examples if set_name == 'train' else db.path_test_examples
		sld_dump_strs = []
		sld_graphs    = []

		with open(examples_path) as f:
			for idx, ppr_example in enumerate(f.readlines()):
				# if idx % 10 == 0:
				# 	print idx
				array = ppr_example.strip().split('\t')
				query = array[0] # neeed to parse it

				positive_examples = [ example[1:] for example in array[1:] if example.strip()[0] == '+']
				negative_examples = [ example[1:] for example in array[1:] if example.strip()[0] == '-']

				positive_substitutions = []
				for positive_example in positive_examples:
					positive_substitution = unification.unify(Atom(query), Atom(positive_example))
					positive_substitutions.append(positive_substitution)

				negative_substitutions = []
				for negative_example in negative_examples:
					negative_substitution = unification.unify(Atom(query), Atom(negative_example))
					negative_substitutions.append(negative_substitution)

				sld_graph = Sld(Atom(query), db, positive_substitutions, negative_substitutions, set_name )
				
				while True:
					took_step = sld_graph.step()
					if not took_step:
						break

				sld_graph.add_back_edges()
				sld_graphs.append(sld_graph)


		for sld_graph in sld_graphs:
			sld_dump_str = sld_graph.dump()
			sld_dump_strs.append(sld_dump_str)


	feature_index_path = os.path.join(processed_data_dir, program_name + '.features' )
	with open(feature_index_path, 'w') as f:		
		f.write('\n'.join(Sld.feature2idx.keys()))




