from equipment_recommender_rag.equipment_inventory.equipment_database import (
    find_equipment_matches,
)

matches = find_equipment_matches(
    equipment_name="transmission grating spectrometer",
    aliases=["TGS", "spectrometer"],
    db_path="data/processed/equipment_inventory.sqlite",
)

for match in matches:
    print(match["canonical_name"], match["match_type"], match["mention_count"])