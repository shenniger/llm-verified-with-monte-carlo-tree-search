import ppo

from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo

from dafny import score_func, can_be_solution

from prompts import prompt, expansion_count, min_lines, check_fun

class GenNode:
    def __init__(self, text, gens):
        self.text = text
        self.gens = gens

montecarlo = MonteCarlo(Node(GenNode(prompt, [])))

def generate_complete(text, montecarlo, gens):
    (text, gen) = ppo.generate(text)
    gens.append(gen)
    score = score_func(text)
    if score is not None:
        if score < 0:
            return None
        else:
            if can_be_solution(text, min_lines, check_fun):
                montecarlo.solution = text
            return GenNode(text, gens)
    else:
        return generate_complete(text, montecarlo, gens)

def child_finder(node, montecarlo):
    state = generate_complete(node.state.text, montecarlo, [])
    if state is None:
        node.update_win_value(-1)
    else:
        child = Node(state)
        node.add_child(child)
        child.update_win_value(1)
        child.update_policy_value(1)

        child = Node(node.state)
        node.add_child(child)
        child.update_policy_value(0.2)

montecarlo.child_finder = child_finder

montecarlo.simulate(expansion_count)

print('CHOSEN SOLUTION')
print(montecarlo.solution)
