# AlexNet Training & Top-5 Error Analysis on Mini-ImageNet

## Problem Statement
Download an implementation of the AlexNet architecture from a neural network library of your choice. Train the network on subsets of varying size from the ImageNet data, and plot the top-5 error with data size.

## Dataset

### Overview
This project uses the **mini-ImageNet dataset**, a curated subset of the full ImageNet dataset specifically designed for efficient deep learning research. Mini-ImageNet provides a balance between computational feasibility and realistic image classification challenges.

**Dataset Composition:**
- **Total Classes:** 100 classes (representative categories from full ImageNet)
- **Image Resolution:** Variable sizes (not fixed at 84×84 as initially assumed)
- **Training Split:** 50,000 images (500 samples per class), balanced distribution
- **Validation Split:** 10,000 images (100 samples per class), balanced distribution
- **Test Split:** 5,000 images (50 samples per class), balanced distribution
- **Source:** Hugging Face Hub (`timm/mini-imagenet`)

**Image Size Distribution:**
The images in mini-ImageNet have **varying dimensions** across all splits:
- **Train Set:** Contains 73+ different image sizes ranging from (307×299) to (800×533), with (500×375) and (500×400) being most common
- **Validation Set:** Contains 55+ different image sizes with similar range, most common being (500×375) and (500×400)
- **Test Set:** Contains 60+ different image sizes with comparable distribution
- **Preprocessing:** All images must be resized to **227×227 pixels** during preprocessing (AlexNet's original input size) before feeding to the model

### Why Mini-ImageNet?
1. **Computational Efficiency:** Smaller images (84×84) and fewer classes (100 vs 1,000) enable faster training
2. **Balanced Distribution:** Equal samples per class prevent bias in learning curve analysis
3. **Research Standard:** Widely used for few-shot learning and data scaling experiments
4. **Realistic Challenges:** Maintains visual complexity and diversity of full ImageNet

### Key Metrics
- **Top-5 Error:** Percentage of samples where the correct label is NOT among the model's top-5 most confident predictions. This metric is historically important from the ImageNet challenge era and allows models to be "slightly wrong" in a controlled way.

---

## Execution Plan

### Overview
Build a modular PyTorch pipeline to train AlexNet on mini-ImageNet with varying dataset sizes, tracking top-5 error across train/val/test splits. Structure code as reusable `.py` modules for training and evaluation logic, save checkpoints and metrics during training, then visualize results in a notebook.

### Phase 1: Dataset Analysis & Preparation

**Goal:** Confirm dataset statistics and design subset sampling strategy

**Tasks:**
1. Execute EDA cells in `eda.ipynb` to verify:
   - 100 classes with ~600 train/~100 val/~100 test samples per class
   - Balanced class distribution across all splits
   - Image dimensions and data format
   
2. Design subset sampling strategy:
   - Use **geometric progression:** 100%, 50%, 25%, 12.5%, 6.25% of full training set
   - Apply **stratified sampling** to preserve class balance in each subset
   - Document subset compositions (e.g., 6.25% = ~3,125 images total = ~31 per class)

**Output:** Dataset statistics documented, sampling logic designed

---

### Phase 2: Modular Code Architecture

**Goal:** Design a clean, maintainable project structure following PyTorch best practices

**Directory Structure:**
```
project/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Mini-ImageNet Dataset class with subset sampling
│   │   ├── transforms.py       # Data augmentation & preprocessing pipelines
│   │   └── dataloader.py       # DataLoader utilities (train/val/test)
│   ├── models/
│   │   ├── __init__.py
│   │   └── alexnet.py          # AlexNet adapted for 84×84 input, 100 classes
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop with validation
│   │   ├── validator.py        # Validation logic & metric computation
│   │   └── checkpoint.py       # Model/optimizer saving & loading
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py        # Test set evaluation pipeline
│   │   ├── metrics.py          # Top-1/top-5 error calculations
│   │   └── results_handler.py  # Results export (JSON/CSV)
│   ├── config/
│   │   ├── __init__.py
│   │   └── default_config.py   # Hyperparameters, paths, subset sizes
   ├── utils/
│       ├── __init__.py
│       ├── metrics.py          # Centralized metric computation (top-1/5 error)
│       └── device.py           # GPU/CPU device management
├── train.py                # Entry point: orchestrate training pipeline
├── eval.py                 # Entry point: run final evaluation
├── eda.ipynb               # Dataset exploratory analysis
├── results.ipynb           # Visualization & result plotting
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

**Design Principles:**
- **Separation of Concerns:** Data loading, model, training, and evaluation are independent modules
- **Configuration Management:** All hyperparameters in `config/default_config.py` (no magic numbers in code)
- **Artifact Tracking:** Save training artifacts (checkpoints, logs) to timestamped directories
- **Reproducibility:** Fixed random seeds, logged hyperparameters, version tracking

---

### Phase 3: Data Pipeline Implementation

**Goal:** Build robust, flexible data loading with subset sampling

**Tasks:**

1. **`data/dataset.py` – Mini-ImageNet Dataset Class**
   - Load dataset from Hugging Face Hub
   - Implement stratified subset sampling (preserve class distribution)
   - Support dynamic subset size specification (percentage or count)
   - Return images + labels with class metadata

2. **`data/transforms.py` – Data Augmentation**
   - **Training:** Resize to 227×227 (AlexNet requirement), random crop, horizontal flip, normalize
   - **Validation/Test:** Resize to 227×227, center crop, normalize (no augmentation)
   - Use consistent statistics (ImageNet mean/std) for normalization

3. **`data/dataloader.py` – DataLoader Utilities**
   - Factory functions: `get_train_loader()`, `get_val_loader()`, `get_test_loader()`
   - Handle batch size, shuffling, and worker processes
   - Ensure validation/test loaders use deterministic ordering

**Output:** Reusable data pipeline with verified class balance preservation

---

### Phase 4: Model Architecture

**Goal:** Implement AlexNet adapted for mini-ImageNet specifications

**Tasks:**

1. **`models/alexnet.py` – AlexNet Implementation**
   - Input size: **227×227 pixels** (original AlexNet specification)
   - Architecture: 5 conv layers + 3 FC layers
   - Output: 100-way softmax (mini-ImageNet classes)
   - Regularization: Dropout (p=0.5 in FC layers)
   - Note: Input resizing handled in `data/transforms.py` before model receives images

2. **Hyperparameter Defaults:**
   - Conv filters: [64, 192, 384, 256, 256]
   - Kernel sizes: [11, 5, 3, 3, 3]
   - Pooling: Max pooling with stride 2
   - FC hidden units: 4096

**Output:** Trainable AlexNet model with consistent 100-class output

---

### Phase 5: Training Pipeline

**Goal:** Implement robust training loop with validation & checkpointing

**Tasks:**

1. **`training/trainer.py` – Main Training Loop**
   - For each epoch:
     - Forward pass on training batch
     - Compute cross-entropy loss
     - Backward pass & optimizer step
     - Log training metrics (loss, top-1 acc, top-5 acc)
     - Run validation after each epoch
     - Save checkpoint if validation loss improves
   - Periodic metrics saving: save metrics to CSV every N epochs (default 10)
   - Final metrics export: save complete metrics history to CSV at end of training
   - Support resuming from checkpoints

2. **`training/validator.py` – Validation Logic**
   - No gradient computation (eval mode)
   - Compute top-1 and top-5 accuracy on full validation set
   - Return metrics dictionary

3. **`training/checkpoint.py` – Checkpoint Management**
   - Save: model state, optimizer state, epoch number, best metrics
   - Load: restore training state for resumption
   - Directory structure: `checkpoints/{experiment_id}/{subset_size}/`
   - CSV metrics export: append new rows efficiently without rewriting entire file

**Output:** Complete training infrastructure with automatic checkpointing and metrics tracking

---

### Phase 6: Evaluation Pipeline

**Goal:** Implement test-time evaluation and results collection

**Tasks:**

1. **`evaluation/evaluator.py` – Test Evaluator**
   - Load best checkpoint for each subset size
   - Run inference on full test set (no gradient computation)
   - Collect predictions and ground truth
   - Save raw results for post-analysis

2. **`utils/metrics.py` – Centralized Metric Computation**
   - Implement `MetricsComputer` class with metric computation methods
   - Implement `compute_top_k_error(predictions, targets, topk=(1,5))` function
   - Compute: top-1 accuracy, top-5 error
   - Return metrics dictionary with keys: `top-1-error`, `top-5-error`, `top-1-accuracy`, `top-5-accuracy`

3. **`evaluation/results_handler.py` – Results Export**
   - Collect metrics across all subset sizes: `{subset_size: {top_1_acc, top_5_error, loss}}`
   - Export to JSON: `results/{experiment_id}/metrics.json`
   - Generate summary: dataset size vs. top-5 error for plotting

**Output:** Comprehensive test metrics across all subset sizes

---

### Phase 7: Configuration & Entry Points

**Goal:** Create flexible configuration system and user-friendly entry scripts

**Tasks:**

1. **`config/default_config.py` – Configuration**
   ```python
   CONFIG = {
       'model': {'name': 'alexnet', 'num_classes': 100, 'dropout': 0.5},
       'training': {'lr': 0.01, 'batch_size': 128, 'epochs': 90, 'momentum': 0.9},
       'subset_sizes': [1.0, 0.5, 0.25, 0.125, 0.0625],  # geometric progression
       'seed': 42,
       'device': 'cuda' if torch.cuda.is_available() else 'cpu'
   }
   ```

2. **`train.py` – Training Entry Point**
   - Parse arguments (subset sizes, learning rate, etc.)
   - For each subset size:
     - Load dataset with specified subset
     - Initialize fresh model
     - Train for fixed epochs
     - Save checkpoints to timestamped directory
   - Log all configurations and results

3. **`eval.py` – Evaluation Entry Point**
   - For each subset size:
     - Load best checkpoint
     - Evaluate on test set
     - Collect metrics
   - Export results to JSON

**Output:** User-friendly scripts: `python train.py` and `python eval.py`

---

### Phase 8: Visualization & Analysis

**Goal:** Create publication-quality plots of learning curves

**Tasks:**

1. **`results.ipynb` – Results Notebook**
   - Load results JSON from evaluation phase
   - **Plot 1:** Top-5 error vs. dataset size (log-log scale, with error bars)
   - **Plot 2:** Training loss & validation top-5 error curves for largest subset
   - **Plot 3:** Comparison table (subset size → final metrics)
   - Save plots as PNG for reporting

2. **Analysis:**
   - Identify scaling law: does error follow power-law ($error \propto size^{-\alpha}$)?
   - Quantify sample efficiency improvements across subsets
   - Discuss convergence behavior with limited data

**Output:** Publication-ready visualizations and insights

---

## Implementation Sequence

1. ✅ **Step 1:** Execute EDA notebook → document dataset statistics
2. ✅ **Step 2:** Create directory structure and config system
3. ✅ **Step 3:** Implement data pipeline (dataset + loaders + transforms)
4. ✅ **Step 4:** Implement AlexNet model
5. ✅ **Step 5:** Implement training loop (trainer + validator + checkpointing)
6. ✅ **Step 6:** Implement evaluation pipeline (evaluator + metrics)
7. ✅ **Step 7:** Create entry scripts (train.py, eval.py)
8. ✅ **Step 8:** Run full pipeline: train on all subset sizes
9. ✅ **Step 9:** Run evaluation on all subset sizes
10. ✅ **Step 10:** Create visualization notebook

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Subset Sampling** | Geometric progression (100%, 50%, 25%, 12.5%, 6.25%) | Reveals learning behavior across multiple regimes |
| **Stratified Sampling** | Yes, preserve class distribution | Ensures fair comparison across subset sizes |
| **Model** | AlexNet (as specified, optional ResNet50 comparison) | Meets requirements; documented baseline from ImageNet era |
| **Hyperparameters** | Fixed across all subsets | Fair comparison; isolate effect of data size |
| **Input Size** | 227×227 pixels (AlexNet original) | Consistent with 2012 ImageNet AlexNet; resizing done in data pipeline |
| **Epochs** | Fixed at 90 | Standard for ImageNet training; prevents overfitting on smaller subsets |
| **Checkpointing** | Save best validation model | Ensure reproducibility; recover best performance |
| **Results Format** | JSON with subset size → metrics | Easy to load and plot in notebook |

---

## Expected Outcomes

- **Code:** Modular, documented, reproducible PyTorch pipeline
- **Data:** 5 training runs (one per subset size) with complete logs
- **Metrics:** Top-5 error curves showing sample efficiency (likely: larger datasets → lower error)
- **Plots:** Professional visualizations for presentation/paper
- **Insights:** Quantified relationship between training set size and generalization error
