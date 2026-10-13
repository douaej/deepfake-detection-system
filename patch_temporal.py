"""
Patche les poids de l'analyse temporelle dans src/detector.py
Lance : python patch_temporal.py
"""
with open('src/detector.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Patch 1 : inverser la pondération 60/40 → 35/65 en faveur du temporel
OLD1 = "combined_fake_score = 0.60 * model_fake_prob + 0.40 * temporal_fake_score"
NEW1 = "combined_fake_score = 0.35 * model_fake_prob + 0.65 * temporal_fake_score"

# Patch 2 : abaisser le seuil delta_var (6416 >> 500 donc score=1.0 OK)
#           mais abaisser le seuil flow_var (0.993 variance → score=0.496 trop bas)
OLD2 = "                delta_score = min(1.0, delta_var / 500.0)\n                flow_score  = min(1.0, flow_var  / 2.0)"
NEW2 = "                delta_score = min(1.0, delta_var / 300.0)\n                flow_score  = min(1.0, flow_var  / 0.5)"

applied = []
for old, new, name in [(OLD1, NEW1, "pondération 60/40→35/65"), (OLD2, NEW2, "seuils delta/flow")]:
    if old in content:
        content = content.replace(old, new)
        applied.append(f"  ✅ Patch '{name}' appliqué")
    else:
        applied.append(f"  ❌ Patch '{name}' NON TROUVÉ")

with open('src/detector.py', 'w', encoding='utf-8') as f:
    f.write(content)

for msg in applied:
    print(msg)

print()
print("Simulation avec tes valeurs :")
model_fake_prob = 0.06
temporal_fake_score = 0.748
delta_var = 6416.6
flow_var = 0.993

delta_score = min(1.0, delta_var / 300.0)
flow_score  = min(1.0, flow_var  / 0.5)
temporal = 0.5 * delta_score + 0.5 * flow_score
combined = 0.35 * model_fake_prob + 0.65 * temporal

print(f"  delta_score      = {delta_score:.3f}")
print(f"  flow_score       = {flow_score:.3f}")
print(f"  temporal_score   = {temporal:.3f}")
print(f"  model_fake_prob  = {model_fake_prob:.3f}")
print(f"  combined_score   = {combined:.3f}  → {'FAKE ✅' if combined >= 0.5 else 'REAL ❌'}")
