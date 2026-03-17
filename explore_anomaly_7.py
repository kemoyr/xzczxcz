import pandas as pd

interactions = pd.read_csv("data/interactions.csv", parse_dates=["event_ts"])

pseudo_start = pd.Timestamp("2025-08-01")
pseudo_end = pd.Timestamp("2025-09-01")
pseudo_post_end = pd.Timestamp("2025-10-01")

pseudo_window = interactions[(interactions["event_ts"] >= pseudo_start) & (interactions["event_ts"] < pseudo_end)]
pseudo_post = interactions[(interactions["event_ts"] >= pseudo_end) & (interactions["event_ts"] < pseudo_post_end)]

pseudo_wishlists = pseudo_window[pseudo_window["event_type"] == 1]
pseudo_reads = pseudo_post[pseudo_post["event_type"] == 2]

wishlist_pairs = set(zip(pseudo_wishlists["user_id"], pseudo_wishlists["edition_id"]))
read_pairs = set(zip(pseudo_reads["user_id"], pseudo_reads["edition_id"]))

overlap = wishlist_pairs & read_pairs
print(f"Aug wishlist pairs: {len(wishlist_pairs)}")
print(f"Sept read pairs: {len(read_pairs)}")
print(f"Overlap (wishlisted in Aug, read in Sept): {len(overlap)}")
