
from pracmln import MLN

mln = MLN(grammar='StandardGrammar')
mln.read('Exogenknowledge/Knowledge.mln')

evidence = {
    'friends': [('a', 'b')],
    'enemy': [('a', 'c')]
}

grounded_mln = mln.ground(evidence=evidence)

query = {
    'friends': [('b', 'c')]
}

result = grounded_mln.infer(query)
print(result)