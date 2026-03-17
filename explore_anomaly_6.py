import pandas as pd

interactions = pd.read_csv("data/interactions.csv", parse_dates=["event_ts"])

# Let's define a clean pseudo-incident window: Sept 2025
pseudo_start = pd.Timestamp("2025-09-01")
pseudo_end = pd.Timestamp("2025-10-01")
pseudo_post_end = pd.Timestamp("2025-11-01")

pseudo_window = interactions[(interactions["event_ts"] >= pseudo_start) & (interactions["event_ts"] < pseudo_end)]
pseudo_post = interactions[(interactions["event_ts"] >= pseudo_end) & (interactions["event_ts"] < pseudo_post_end)]

pseudo_wishlists = pseudo_window[pseudo_window["event_type"] == 1]

# How many wishlist items from Sept were read/interacted with in Oct?
wishlist_pairs = set(zip(pseudo_wishlists["user_id"], pseudo_wishlists["edition_id"]))
post_pairs = set(zip(pseudo_post["user_id"], pseudo_post["edition_id"]))

overlap = wishlist_pairs & post_pairs
print(f"Sept wishlist pairs: {len(wishlist_pairs)}")
print(f"Oct post pairs: {len(post_pairs)}")
print(f"Overlap (wishlisted in Sept, interacted in Oct): {len(overlap)}")

# Also consider deduplication nature: what if "lost" items are those that have multiple events in a short time?
# Or what if "lost" items are just the ones that appear in the post-incident window?
