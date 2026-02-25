import plan_cache_engine
from finbench_utils import get_questions
import matplotlib.pyplot as plt

test_engine = plan_cache_engine()
questions = get_questions()

coses = []
for q in questions:
    _, max_cosine = test_engine.retrieve_plan(q)
    coses.append(max_cosine)

plt.scatter(coses)
plt.show()
