from pathlib import Path
import sys

from inference.predict_image import predict_attributes
from agents.facial_analysis_agent import analyse_face

# =====================================================
# IMAGE ARGUMENT
# =====================================================

if len(sys.argv) != 2:

    print(
        "Usage: python src/test_agent.py image.jpg"
    )

    exit()

image_path = sys.argv[1]

# =====================================================
# PREDICT
# =====================================================

results = predict_attributes(
    image_path
)

print("\nPredictions")

for attribute, value in results.items():

    print(
        f"{attribute}: {value:.4f}"
    )

# =====================================================
# AGENT ANALYSIS
# =====================================================

analysis = analyse_face(
    results
)

print("\nAgent Analysis\n")

print(
    analysis
)