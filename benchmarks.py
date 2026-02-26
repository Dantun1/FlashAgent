from plan_cache_engine import PlanCacheEngine, AgentBlueprint
from utils.finbench_utils import get_questions
import matplotlib.pyplot as plt

test_engine = PlanCacheEngine()
questions = get_questions()

prev_hits = 0
for q in questions:
    test_engine.retrieve_plan(q)

    hits, total, ratio = test_engine.hit_stats
    if hits == prev_hits:
        print(f"CACHE MISS")
    else:
        print(f"CACHE HIT")

    print(f"\nCURRENT SCORE: {ratio}")

    prev_hits = hits
    


sims = test_engine.similarities
plt.scatter(range(len(sims)),sims)
plt.show()
