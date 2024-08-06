'''
causal-learn has a way to inject background knowledge (whether that knowledge be incomplete or complete)
Run this file to observe how the two main functionalities work - there are many other restrictions/constraints we can apply
For the purposes of demonstration, the two functionalities shown are where an edge is required and forbidden
'''

import numpy as np, pandas as pd
import dowhy
import dowhy.gcm as gcm
from dowhy import CausalModel
import networkx as nx
from causallearn.search.ConstraintBased.PC import pc
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

X = np.random.normal(loc=0, scale=1, size=10000)
Y = 2 * np.random.normal(loc=0, scale=1, size=10000)
Z = 3 * Y + np.random.normal(loc=0, scale=1, size=10000)
B = 5 * Y + 2 * Z + np.random.normal(loc=0, scale=1, size=10000)
R = 7 * X + 3 * Z + 5 * Y + np.random.normal(loc=0, scale=1, size=10000)
data = pd.DataFrame(data=dict(X=X, Y=Y, Z=Z, B=B, R=R))

dgp = nx.DiGraph([('X', 'R'), ('Y', 'Z'), ('Y', 'B'), ('Y', 'R'), ('Z', 'B'), ('Z', 'R')])
causal_model = gcm.StructuralCausalModel(dgp)
gcm.auto.assign_causal_mechanisms(causal_model, data)
gcm.fit(causal_model, data)

generated_data = gcm.draw_samples(causal_model, num_samples=100000)
labels = ['X', 'Y', 'Z', 'B', 'R']

print("Generated Dataset: ")
print(generated_data.head())
print()

data = generated_data
data = data.to_numpy()

cg = pc(data=data, show_progress=False, node_names=labels)
# cg.draw_pydot_graph()

nodes = cg.G.get_nodes()

bk = BackgroundKnowledge().add_required_by_node(nodes[0], nodes[1])

cg_w_bk = pc(data=data, show_progress=False, node_names=labels, background_knowledge=bk)
print(cg_w_bk.G.get_graph_edges()[6])

# cg_w_bk.draw_pydot_graph()




