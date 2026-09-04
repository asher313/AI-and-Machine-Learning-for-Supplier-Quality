# Chapter 6 — 6.5 Inspect First, Always
df.shape            # (rows, columns)
df.head()           # first 5 rows — do the values look sane?
df.info()           # dtypes, non-null counts, memory
df.describe()       # min/max/quartiles for numeric columns
df.isnull().sum()   # missing values per column
df.duplicated().sum()  # exact duplicate rows
df.nunique()        # distinct values per column
