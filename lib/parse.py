import re
from copy import deepcopy
from collections import defaultdict
import os

class Symbol:
    def __init__ (self, name):
        self.name = name
            
    def __str__(self):
        return self.name

class Constant(Symbol):
    def value(self): 
        return self.name

class Variable(Symbol):
    pass

class Relation(Symbol):
    pass

def symbol(name):
    if (name[0] in name.upper()):
        return Variable(name)
    else:
        return Constant(name)

# !! Hey atom could be constant as well!!! Right?
class Atom:

    # def __init__ (self, relation, arguments):
    #     self.relation = relation
    #     self.arguments = arguments

    def __init__(self, atom_text):
        array = atom_text.strip().replace('(', ',').replace(')', '').split(',')
        relation = Relation(array[0])
        argument_texts = array[1:]
        arguments = []
        for argument_text in argument_texts:
            arguments.append( symbol(argument_text) )
        self.relation = relation
        self.arguments = arguments

    def rename(self):
        atom = deepcopy(self)
        for argument in atom.arguments:
            argument.name = argument.name + '_' if isinstance(argument, Variable) else argument.name
        return atom

    def apply_substitution(self, substitution):
        atom = deepcopy(self)
        for argument in atom.arguments:
            if argument.name == substitution.keys():
                argument.name = substitution[argument.name]
        return atom

    def __str__(self):
        return ''.join([str(self.relation), '(', ','.join([ str(arg) for arg in self.arguments]), ')'])

    def __len__(self):
        return len(self.arguments)

# this doesn't handle functors yet
def parse_atom(atom_str):
    match = re.match( '(.+)\((.+)\)', atom_str )
    relation_name = match[1]
    arguments_str = match[2]
    arguments = arguments_str.split(',')    
    relation  = Relation(relation_name)
    def symbol(name):
        return Variable(name) if (name[0] in string.uppercase) else Constant(name)
    arguments = [ symbol(argument) for argument in arguments]
    return Atom(relation, arguments)


class Database:

    def __init__(self, name, input_dir):
        self.rules = []
        self.name  = name
        self.input_dir = input_dir

        self.path_ppr       = os.path.join(self.input_dir, self.name, self.name + '.ppr')
        self.path_cfacts    = os.path.join(self.input_dir, self.name, self.name + '.cfacts')
        self.path_graph     = os.path.join(self.input_dir, self.name, self.name + '.graph')
        self.path_train_examples  = os.path.join(self.input_dir, self.name, self.name + '-train.examples')
        self.path_test_examples   = os.path.join(self.input_dir, self.name, self.name + '-test.examples')

    def insert_rule(self, head, body, features):
        self.rules.append((head, body, features))


    @staticmethod
    def rename_rule(rule):

        head, body, features = rule
        head = head.rename()
        body = [ atom.rename() for atom in body]
        features = [ feature.rename() for feature in features]
        return [head, body, features]

    def build_index(self):
        db_index = defaultdict(set)
        for idx, rule in enumerate(self.rules):
            head_atom = rule[0]
            ground_tokens = [head_atom.relation.name]
            for argument in head_atom.arguments:
                if argument.name[0].islower():
                    ground_tokens.append( argument.name )
            for token in ground_tokens:
                db_index[token].add(idx)
        self.db_index = db_index

    def get_rules_with_tokens(self, tokens):
        rule_idxs = set.intersection(*[ self.db_index[token] for token in tokens])
        return [ self.rules[rule_idx] for rule_idx in rule_idxs]


    def load_ppr(self):

        with open(self.path_ppr) as f:
            for line in f.readlines():
                if line.strip().startswith('#') or line.strip() == '':
                    continue
                line = line.strip()
                if ':-' in line:
                    head, body_parts = line.strip().split(':-')
                else:
                    head = line.strip()
                    body_parts = ''
                head = head.strip()
                body_parts = body_parts.strip()
                if '{' in body_parts and '}' in body_parts:
                    match = re.match( '([^{}]+)?({.+})?', body_parts.strip() )
                    body = match.group(1)
                    if not body:
                        body_atoms = []
                    else:
                        body = body.replace(".", '').replace(' ', '')
                        body_atoms = body.replace('),', ');').split(';')
                    features = match.group(2)
                    feature_atoms = tuple(features.replace(' ', '').replace('{', '').replace('}', '').split(':'))
                    head_atom = head.strip()
                elif '#' in body_parts:
                    match = re.match( '(.+)#(.+)', body_parts.strip() )
                    body = match.group(1)
                    if not body:
                        body_atoms = []
                    else:
                        body = body.replace(".", '').replace(' ', '')
                        body_atoms = body.replace('),', ');').split(';')
                    features = match.group(2)
                    feature_atoms = tuple([features, '1.0'])
                    head_atom = head.strip()
                else:
                    body = body_parts
                    if not body:
                        body_atoms = []
                    else:
                        body = body.replace(".", '').replace(' ', '')
                        body_atoms = body.replace('),', ');').split(';')
                    head_atom = head.strip()
                    feature_atoms = ()

                head_atom = Atom(head_atom)
                body_atoms = [ Atom(atom) for atom in body_atoms]


                # feature_atoms would always be a tuple ()
                # of 0 or 1 or 2 elements
                # the elements can be simple strings or even atoms ...
                if len(feature_atoms) == 0:
                    feature_atoms = []
                elif len(feature_atoms) == 1:
                    feature_atoms = [Atom(feature_atoms[0])]
                else:
                    feature_atoms = [Atom(feature_atoms[0]), Atom(feature_atoms[1])]

                self.insert_rule( head_atom, body_atoms, feature_atoms)

    def load_cfacts(self):

        with open(self.path_cfacts) as f:
            for line in f.readlines():
                array = line.strip().split('\t')
                relation = array[0]
                arguments = array[1:]
                atom_str = relation+'('+','.join(arguments)+')'
                head = Atom( atom_str )
                self.insert_rule( head, [], [] )

        
    def load_graph(self):
        with open(self.path_graph) as f:
            for line in f.readlines():
                array = line.strip().split('\t')
                relation = array[0]
                arguments = array[1:]
                atom_str = relation+'('+','.join(arguments)+')'
                head = Atom( atom_str )
                self.insert_rule( head, [], [] )



########################################

# db_name = 'sample'
# db = Database('sample')

# with open(db.name + ".ppr") as f:
#     for line in f.readlines():
#         if line.strip().startswith('#') or line.strip() == '':
#             continue
#         head, body_parts = line.strip().split(':-')
#         head = head.strip()
#         body_parts = body_parts.strip()
#         if '{' in body_parts and '}' in body_parts:
#             match = re.match( '([^{}]+)?({.+})?', body_parts.strip() )
#             body = match.group(1)
#             if not body:
#                 body_atoms = []
#             else:
#                 body = body.replace(".", '').replace(' ', '')
#                 body_atoms = body.replace('),', ');').split(';')
#             features = match.group(2)
#             feature_atoms = tuple(features.replace(' ', '').replace('{', '').replace('}', '').split(':'))
#             head_atom = head.strip()
#         elif '#' in body_parts:
#             match = re.match( '(.+)#(.+)', body_parts.strip() )
#             body = match.group(1)
#             if not body:
#                 body_atoms = []
#             else:
#                 body = body.replace(".", '').replace(' ', '')
#                 body_atoms = body.replace('),', ');').split(';')
#             features = match.group(2)
#             feature_atoms = tuple([features, '1.0'])
#             head_atom = head.strip()
#         else:
#             body = body_parts
#             if not body:
#                 body_atoms = []
#             else:
#                 body = body.replace(".", '').replace(' ', '')
#                 body_atoms = body.replace('),', ');').split(';')
#             head_atom = head.strip()
#             feature_atoms = ()

#         print '-----------------'
#         print line.strip()
#         print '-----------------'
#         # print 'head_atom:'
#         print head_atom
#         # print 'body_atoms:'
#         print body_atoms
#         # print 'feature_atoms:'
#         print feature_atoms

#         head_atom = Atom(head_atom)
#         body_atoms = [ Atom(atom) for atom in body_atoms]
#         feature_atoms = [ Atom(atom) for atom in feature_atoms]

#         db.insert_rule( head_atom, body_atoms, feature_atoms)

# for ext in [".cfacts", ".graph"]:
#     with open( db.name + ext ) as f:
#         for line in f.readlines():
#             array = f.split('   ')
#             relation = array[0]
#             arguments = array[1:]
#             head = Atom( Symbol(relation), [ Constant(argument) for argument in arguments] )
#             db.insert_rule( head, [], [] )

# #########################################
# #########################################






# ToDo Next:
# 1. How to check that solution substitution is actually correct or incorrect
# 0. Lets leave it for now and try to see if other things work. Label adding can be done later!







