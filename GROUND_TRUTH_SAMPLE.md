# Ground Truth — Representative Questions & Answers

Ground truth answers are deterministic counts or feature identifiers computed by executing the correct query against the source data. Each entry includes the list of specific feature IDs that constitute the answer, ensuring reproducibility. The LLM judge evaluates whether the system's natural language response correctly conveys this factual answer.

---

## spatial_radius (43 questions) — "How many X within Y distance of Z?"

**q049:** "I'm doing a traffic signage audit around Brønshøj Torv. How many road signs are registered within 100 metres of the square?"
- Expected answer: `1`
- Dataset: `traffic_signs_including_removed_signs_koebenhavn`
- Spatial: (55.708, 12.507), radius 100m
- Verified feature IDs: `[441e1d3a-dee4-4687-b00a-eee8a100bcbf]`

**q105:** "I want to lock my bike and walk across Dronning Louises Bro. Are there bike racks right at the bridge entrance?"
- Expected answer: `2`
- Dataset: `bicycle_parking_racks_koebenhavn_kommune`
- Spatial: (55.6871, 12.5638), radius 100m
- Verified feature IDs: `[4892, 4893]`

---

## spatial_nearest (19 questions) — "What is the closest X to Y?"

**q128:** "I live in Risskov in northern Aarhus and want to rent an allotment garden. Which allotment association is closest to my neighbourhood?"
- Expected answer: `8c2d727f-2b80-41a7-a2d4-6661b5cf17d1` (feature ID)
- Dataset: `allotment_gardens_aarhus_kommune`
- Spatial center: (56.1895, 10.21)

**q129:** "It's a hot summer day and I'm outside Tivoli with my kids. Where is the nearest drinking water fountain?"
- Expected answer: `c9fd7e53-239e-453b-bcf5-cebd06f62af7` (feature ID)
- Dataset: `drinking_water_posts_and_fountains_koebenhavn`
- Spatial center: (55.6738, 12.5681)

---

## property_filter (44 questions) — "How many X with property=value?"

**q062:** "I'm a Nørrebro resident and need to drop off some household recycling. How many local recycling stations are there in the area?"
- Expected answer: `4`
- Dataset: `recycling_centres_koebenhavn_kommune`
- Verified feature IDs: `[8, 7, 24, 32]`

**q055:** "I'm looking for a municipal council-run daycare near Østerbro — we prefer publicly operated nurseries over private ones. Are there any nearby?"
- Expected answer: `1`
- Dataset: `daycare_institutions_koebenhavn_kommune`
- Filter: `ownership` field
- Verified feature IDs: `[1319]`

---

## conversational (30 questions) — informal/vague language, same as spatial_radius

**q068:** "I need somewhere to sit down near Tivoli. Anything?"
- Expected answer: `4`
- Dataset: `city_furniture_inventory_koebenhavn`
- Spatial: (55.6737, 12.5681), radius 150m
- Verified feature IDs: `[064a3af1..., e59b80de..., edf8bf71..., 4c1dbc83...]`

**q074:** "Need a nursery for our 2-year-old near Valby."
- Expected answer: `6`
- Dataset: `daycare_institutions_koebenhavn_kommune`
- Spatial: (55.6638, 12.511), radius 500m

---

## unanswerable (63 questions) — cannot be answered from available data

**q194:** "Are the bicycle parking racks in Copenhagen secure enough to leave my expensive road bike overnight?"
- Expected answer: `NOT_ANSWERABLE`
- Reason: Security/theft risk is not a stored property in any dataset.

**q168:** "I'm cycling around Aarhus and I see some old terraced houses near a park. Are these buildings protected in any way?"
- Expected answer: `NOT_ANSWERABLE`
- Reason: The question is vague ("near a park") and doesn't specify which buildings; heritage data exists for Aarhus but the question cannot be resolved to specific features.
