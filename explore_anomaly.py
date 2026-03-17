import pandas as pd

interactions = pd.read_csv("data/interactions.csv", parse_dates=["event_ts"])
print("Total interactions:", len(interactions))

# Define windows
incident_start = pd.Timestamp("2025-10-01")
incident_end = pd.Timestamp("2025-11-01")
post_end = pd.Timestamp("2025-12-01")

clean = interactions[interactions["event_ts"] < incident_start]
incident = interactions[(interactions["event_ts"] >= incident_start) & (interactions["event_ts"] < incident_end)]
post = interactions[(interactions["event_ts"] >= incident_end) & (interactions["event_ts"] < post_end)]

print("\n--- Interaction Counts by Window ---")
print("Clean (last 30 days of clean):", len(clean[clean["event_ts"] >= incident_start - pd.Timedelta(days=30)]))
print("Incident (31 days):", len(incident))
print("Post-Incident (30 days):", len(post))

print("\n--- Event Type Distribution ---")
print("Clean (last 30 days) event types:\n", clean[clean["event_ts"] >= incident_start - pd.Timedelta(days=30)]["event_type"].value_counts())
print("Incident event types:\n", incident["event_type"].value_counts())
print("Post-Incident event types:\n", post["event_type"].value_counts())

print("\n--- Interaction Counts by Hour ---")
print("Incident hours:\n", incident["event_ts"].dt.hour.value_counts().sort_index())
print("Post-Incident hours:\n", post["event_ts"].dt.hour.value_counts().sort_index())

print("\n--- Users check ---")
targets = pd.read_csv("data/targets.csv")
print("Targets length:", len(targets))
target_users = set(targets["user_id"])

incident_users = set(incident["user_id"])
print("Target users in incident:", len(target_users & incident_users))
print("Target users in post:", len(target_users & set(post["user_id"])))

clean_users = set(clean["user_id"])
print("Target users in clean:", len(target_users & clean_users))

# Let's check event_type missingness specifically for targets
targets_incident = incident[incident["user_id"].isin(target_users)]
print("\nTargets incident event types:\n", targets_incident["event_type"].value_counts())

# What about modulus/parity of user_id or edition_id?
print("\nTargets incident user_id % 2:\n", (targets_incident["user_id"] % 2).value_counts())
print("Targets incident edition_id % 2:\n", (targets_incident["edition_id"] % 2).value_counts())
