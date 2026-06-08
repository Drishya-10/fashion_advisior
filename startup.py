import os
import gdown

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/embedding_vectors.npy"):
    print("Downloading embedding_vectors.npy...")
    gdown.download(
        "https://drive.google.com/uc?id=15X7tBkiK-BXOf_7drKlZexnX8c5pf--Z",
        "data/embedding_vectors.npy",
        quiet=False
    )
    print("✅ Vectors ready")

if not os.path.exists("data/embedding_ids.npy"):
    print("Downloading embedding_ids.npy...")
    gdown.download(
        "https://drive.google.com/uc?id=1Dwwiro8cAIvAWTVvDHe3sUpTahtowttS",
        "data/embedding_ids.npy",
        quiet=False
    )
    print("✅ IDs ready")

print("✅ Startup complete")