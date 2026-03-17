import pandas as pd
users = pd.read_csv("data/users.csv")
print("Total users:", len(users))
print("User ID parity:")
print((users["user_id"] % 2).value_counts())
