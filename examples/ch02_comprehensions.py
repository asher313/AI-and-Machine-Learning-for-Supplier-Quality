# Chapter 2 — Comprehensions
open_ids = [n["ncr_id"] for n in ncrs if n["closed_at"] is None]
suppliers = {n["supplier_id"] for n in ncrs}          # a set
by_id = {n["ncr_id"]: n for n in ncrs}                # a dict
total_qty = sum(n["quantity"] for n in ncrs)          # lazy
