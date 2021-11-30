#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
from sklearn.metrics import r2_score, mean_squared_error

friction_univ = np.linspace(0.1, 0.3, 3)
friction = ctrl.Antecedent(friction_univ, 'friction')
friction.automf(3)

initial_v_univ = np.linspace(1, 5, 3)
initial_v = ctrl.Antecedent(initial_v_univ, 'initial_v')
initial_v.automf(3)

final_x_univ = np.linspace(0.1, 12.8, 7)
final_x = ctrl.Consequent(final_x_univ, 'final_x')
final_x.automf(7)

rules = []
rules.append(ctrl.Rule(initial_v["poor"], final_x["dismal"]))
rules.append(ctrl.Rule(friction["poor"] &
             initial_v["average"], final_x["mediocre"]))
rules.append(ctrl.Rule(friction["poor"] &
             initial_v["good"], final_x["excellent"]))
rules.append(ctrl.Rule(friction["average"] &
             initial_v["average"], final_x["poor"]))
rules.append(ctrl.Rule(friction["average"] &
             initial_v["good"], final_x["average"]))
rules.append(ctrl.Rule(friction["good"] &
             initial_v["average"], final_x["poor"]))
rules.append(ctrl.Rule(friction["good"] &
             initial_v["good"], final_x["mediocre"]))

final_x_ctrl = ctrl.ControlSystem(rules)
final_x_sim = ctrl.ControlSystemSimulation(final_x_ctrl)

f = 0.1
final_x_sim.input["friction"] = f

vs = np.linspace(1, 5, 50)
x_trues = []
x_preds = []
for v in vs:
    final_x_sim.input["initial_v"] = v
    final_x_sim.compute()
    x_true = v * v / (2 * f * 9.81)
    x_pred = final_x_sim.output["final_x"]
    x_trues.append(x_true)
    x_preds.append(x_pred)

print("R2", r2_score(x_trues, x_preds))
print("MSE", mean_squared_error(x_trues, x_preds))

plt.figure(figsize=(9, 7))
plt.plot(x_trues, vs, "bo--")
plt.plot(x_preds, vs, "go--")
plt.title(f"Final position for friction = {f}")
plt.xlabel("Final x")
plt.ylabel("Initial v")
plt.legend(["True values", "Predicted values"])
plt.show()
