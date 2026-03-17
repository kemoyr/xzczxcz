import pandas as pd

interactions = pd.read_csv("data/interactions.csv", parse_dates=["event_ts"])

incident_start = pd.Timestamp("2025-10-01")
incident_end = pd.Timestamp("2025-11-01")

incident = interactions[(interactions["event_ts"] >= incident_start) & (interactions["event_ts"] < incident_end)]
clean = interactions[(interactions["event_ts"] >= incident_start - pd.Timedelta(days=31)) & (interactions["event_ts"] < incident_start)]

print("Incident events by day:")
print(incident["event_ts"].dt.date.value_counts().sort_index())

print("\nIncident events by event_type by day:")
print(incident.groupby([incident["event_ts"].dt.date, "event_type"]).size().unstack())

print("\nClean events by day:")
print(clean["event_ts"].dt.date.value_counts().sort_index())
