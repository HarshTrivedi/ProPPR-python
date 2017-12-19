

# atom:
# P(one,two,X,world)  = P(A,B,hello, H).

# try few examples from lecture notes



from parse import *

# # atoms is list of atoms
def apply_substitution( operands, mgu ):
	# mgu is actually a dict of variable substitutions
	# mgu: key --> Variable Class Object
	# mgu: val --> Variable or Constant Class Object
	# NO ! it should be just string to string match - keyVal pair

	substituted_operands = []
	for operand in operands:
		if isinstance(operand, Atom):
			args = operand.arguments
			subst_args = []
			for arg in args:
				if arg.name in mgu.keys():
					subst_args.append( symbol(mgu[arg.name]) )
				else:
					subst_args.append( arg )
			operand.arguments = subst_args
			substituted_operands.append(operand)
		else:
			if operand.name in mgu.keys():				
				substituted_operands.append( symbol(mgu[operand.name]) )
			else:
				substituted_operands.append( operand )
	return substituted_operands


def is_case(substitutions, pick_index):

	picked_substitution = substitutions[pick_index]
	rest_substitutions  = substitutions[:pick_index]+substitutions[pick_index+1:]
	arg_left, arg_right = picked_substitution

	rest_substitutions_items = []
	for substitution in rest_substitutions:
		items = []
		for e in substitution:
			if isinstance(e, Atom):
				items.append( [ x.name for x in e.arguments] )
			else:
				items.append( e.name )
		rest_substitutions_items.append(items)
	rest_count = len([ 1 for items in rest_substitutions_items if arg_left.name in items])

	if isinstance(arg_left, Atom) and isinstance(arg_right, Atom):
		if arg_left.relation.name == arg_right.relation.name and len(arg_left) == len(arg_right):
			return 1
		else:
			return 2
	if isinstance(arg_left, Constant) and isinstance(arg_right, Constant) and arg_left.name != arg_right.name:
		return 2
	elif arg_left.name == arg_right.name:
		return 3
	elif not isinstance(arg_left, Variable) and isinstance(arg_right, Variable):
		return 4
	elif  isinstance(arg_right, Atom) and arg_left.name in map( lambda x: x.name, arg_right.arguments):
		return 5
	elif isinstance(arg_left, Variable) and rest_count > 0:
		return 6 #5b
	else:
		return 7



# substitutions just has to be tupled list
def unify_step(substitutions):
	if len(substitutions) == 0:
		return None

	case_number = is_case( substitutions, 0 )
	picked_substitution = substitutions[0]
	rest_substitutions  = substitutions[1:]
	arg_left  = picked_substitution[0]
	arg_right = picked_substitution[1]

	if case_number == 1:
		for argL, argR in zip(arg_left.arguments, arg_right.arguments):
			rest_substitutions.append( (argL, argR) )
		return rest_substitutions
	elif case_number == 2:
		return None
	elif case_number == 3:
		return rest_substitutions
	elif case_number == 4:
		rest_substitutions.append( (arg_right, arg_left) )
		return rest_substitutions
	elif case_number == 5:
		return None
	elif case_number == 6:
		updated_rest_substitutions = []
		for x, y in rest_substitutions:
			x_dash = arg_right if x.name == arg_left.name else x
			y_dash = arg_right if y.name == arg_left.name else y
			updated_rest_substitutions.append( (x_dash, y_dash) )
		updated_rest_substitutions.append( (arg_left, arg_right) )
		return updated_rest_substitutions
	else:		
		rest_substitutions.append(picked_substitution)
		return rest_substitutions


def unify(x, y):
	substitutions = [(x,y)]
	while True:
		substitutions = unify_step(substitutions)
		if substitutions is None:
			return None
		else:
			possible_picks = len(substitutions)
			pick_cases = list(set([ is_case(substitutions, pick_index) for pick_index in range(possible_picks)]))
			if len(pick_cases) == 1 and pick_cases[0] == 7:
				# means nothing more is possible! return
				# return dict(substitutions)
				return { x.name: y.name for x, y in substitutions}


# MGU : key has to be a variable
#   and val has to be either atom or constant or another variable
# 
# For now lets assume that they are variables and constants only!
# and so let mgus be string dict of names
def compose_mgus(mgu1, mgu2):
	mgu1_composed = []
	mgu1_keys = set()
	for x, y in mgu1.items():
		y_dash = mgu2[y] if y in mgu2.keys() else y
		mgu1_keys.add(x)		
		if x != y_dash:
			mgu1_composed.append((x, y_dash))
	mgu2_updated = []
	for x, y in mgu2.items():
		if not x in mgu1_keys:
			mgu2_updated.append( (x,y) )
	return dict(mgu1_composed + mgu2_updated)


# def apply_substitution(operand, mgu):
# 	operand


# x = Atom('P(one,two,X,T)')
# y = Atom('P(A,B,hello,three)')


# unification = unify(x, y)
# print '---------------'
# if unification:
# 	for x, y in unification:
# 		print [str(x), str(y)]
# else:
# 	print None



# x = Atom('predict(train00004,Y)')
# y = Atom('predict(X_,Y_)')
# unification = unify(x, y)

# if unification:
# 	for x, y in unification.items():
# 		print [str(x), str(y)]
# else:
# 	print None


# operands = [ Atom('isLabel(Y_)'), Atom('ab_classify(X_,Y_)')]
# x = apply_substitution( operands, {'X_': 'train00004', 'Y': 'Y_'} )
# print [str(x[0]), str(x[1])]







