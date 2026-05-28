import urllib.request
import json
import urllib.parse

BBOX = "5.0604,52.2496,5.9684,52.8065"
limit = 1000
base_url = "https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas/items"

# We will query first 5000 features to see what crop names are available
features_collected = []
url = f"{base_url}?bbox={BBOX}&limit={limit}&f=json"

print("Downloading BRP crop name catalog...")
for page in range(1, 6):
    if not url:
        break
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        curr_features = data.get("features", [])
        if not curr_features:
            break
        features_collected.extend(curr_features)
        
        # Find next link
        links = data.get("links", [])
        next_url = None
        for link in links:
            if link.get("rel") == "next":
                next_url = link.get("href")
                break
        url = next_url
        print(f"  Page {page} done, total features: {len(features_collected)}")
    except Exception as e:
        print("Error on page", page, e)
        break

# Extract all unique crop names
crop_counts = {}
for feat in features_collected:
    props = feat.get("properties", {})
    gewas = props.get("gewas", "Unknown")
    crop_counts[gewas] = crop_counts.get(gewas, 0) + 1

print("\nUnique crop names in sample:")
for k, v in sorted(crop_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  '{k}': {v}")
