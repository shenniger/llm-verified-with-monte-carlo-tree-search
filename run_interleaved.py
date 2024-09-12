from lang import can_be_solution
from lang import score_func as uncached_score_func

from common_cache import create_cached_func
score_func, cache_stats, reset_cache = create_cached_func(uncached_score_func)
from common_interactive import diffprompt, onelineonly

from prompts import prompt, expansion_count, min_lines, check_func
from common import limit_depth, max_completion_depth
from common_stats import stats

import llm
import sys

import common_wandb

import time
init_time = time.time()

def generate_complete(text, current_completion_depth=1):
    if current_completion_depth >= max_completion_depth:
        return None
    prev = text
    texts = llm.generate(text, 1)
    texts = onelineonly(prev, texts)
    text = texts[0]
    score = score_func(text)
    print(diffprompt(prev, texts))
    if score is not None:
        if score < 0:
            return None
        else:
            if can_be_solution(text, min_lines, check_func):
                print("CHOSEN SOLUTION")
                print(text)
                common_wandb.compute_summary_nomc(text, init_time)
                sys.exit(0)
            return text
    else:
        return generate_complete(text, current_completion_depth + 1)

class Node:
    def __init__(self, base, depth):
        self.base = base
        self.branches = []
        self.alive = True
        self.depth = depth

    def generate(self):
        print("[", self.depth, "] Generating more")
        new_text = generate_complete(self.base)
        if new_text is None:
            return False
        self.branches.append(Node(new_text, self.depth + 1))
        return True

    def steps(self):
        for b in self.branches:
            print("[", self.depth, "] Branch")
            b.steps()
        if self.alive:
            if self.generate():
                self.branches[len(self.branches)-1].steps()
            else:
                self.alive = False

def main(prompt = prompt):
    root = Node(prompt, 1)
    while True:
        root.steps()

if __name__ == "__main__":
    main()
