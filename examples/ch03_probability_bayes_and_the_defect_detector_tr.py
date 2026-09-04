# Chapter 3 — 3.5 Probability: Bayes and the Defect-Detector Trap
prevalence = 0.005      # prior: P(defective)
sensitivity = 0.99      # P(flag | defective)
fpr = 0.01              # P(flag | good)

p_flag = sensitivity * prevalence + fpr * (1 - prevalence)
posterior = sensitivity * prevalence / p_flag

print(round(p_flag, 5), round(posterior, 4))
