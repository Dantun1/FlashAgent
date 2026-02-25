from plan_cache_engine import PlanCacheEngine, AgentBlueprint
from finbench_utils import get_questions
import matplotlib.pyplot as plt

test_engine = PlanCacheEngine()
questions = get_questions()


coses = []
for q in questions[10:20]:
    result = test_engine.retrieve_plan(q)

    if isinstance(result, str):  # "DB Empty"
        continue

    _, max_cosine = result
    coses.append(max_cosine)

plt.scatter(range(len(coses)),coses)
plt.show()
