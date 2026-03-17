import pandas as pd

interactions = pd.read_csv("data/interactions.csv", parse_dates=["event_ts"])
targets = pd.read_csv("data/targets.csv")

incident_start = pd.Timestamp("2025-10-01")
incident_end = pd.Timestamp("2025-11-01")

incident = interactions[(interactions["event_ts"] >= incident_start) & (interactions["event_ts"] < incident_end)]

print("Total targets:", len(targets))
print("Target user_id parity:")
print((targets["user_id"] % 2).value_counts())

print("\nAll incident user_id parity:")
print((incident["user_id"] % 2).value_counts())

print("\nAll clean user_id parity (last 30 days):")
clean = interactions[(interactions["event_ts"] >= incident_start - pd.Timedelta(days=30)) & (interactions["event_ts"] < incident_start)]
print((clean["user_id"] % 2).value_counts())

print("\nIs it just user_id % 2 == 1 missing in incident?")
print("Count of odd users in incident:", len(incident[incident["user_id"] % 2 == 1]))
