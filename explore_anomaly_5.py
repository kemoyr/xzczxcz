import pandas as pd

interactions = pd.read_csv("data/interactions.csv", parse_dates=["event_ts"])
targets = pd.read_csv("data/targets.csv")
target_users = set(targets["user_id"])

incident_start = pd.Timestamp("2025-10-01")
incident_end = pd.Timestamp("2025-11-01")

post = interactions[interactions["event_ts"] >= incident_end]
incident = interactions[(interactions["event_ts"] >= incident_start) & (interactions["event_ts"] < incident_end)]

print("Post interactions:", len(post))
print("Target users in post:", len(set(post["user_id"]) & target_users))

# Are there users who have events in Post but not in Incident?
post_users = set(post["user_id"])
incident_users = set(incident["user_id"])
print("Target users in Post but NOT in Incident:", len((post_users - incident_users) & target_users))
