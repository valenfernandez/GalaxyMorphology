galaxy-morphology/
│
├── data/
│   └── Galaxy10_DECals.h5
│
├── src/
│   ├── data/              # data loading & transforms
│   ├── models/            # CNN / ViT architectures
│   ├── training/          # training loops
│   ├── evaluation/        # metrics & plots
│   └── explainability/    # Grad-CAM
│
├── scripts/
│   └── dataset_inspection.py
│
├── experiments/           # saved runs, logs