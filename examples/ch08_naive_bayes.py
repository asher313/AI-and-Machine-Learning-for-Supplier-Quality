# Chapter 8 — 8.6 Naive Bayes
from sklearn.naive_bayes import (
    BernoulliNB, GaussianNB, MultinomialNB,
)

model = GaussianNB()             # continuous features
model = MultinomialNB(alpha=1.0) # word counts
model = BernoulliNB(alpha=1.0)   # binary features
