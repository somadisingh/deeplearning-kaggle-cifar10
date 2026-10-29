# Efficient ResNet on CIFAR-10 (< 5M params)

Kaggle course challenge: train a ResNet-style classifier from scratch on the CIFAR-10
training set (50,000 labelled 32x32 PNGs) with at most 5M trainable parameters, and
predict labels for 2,000 held-out evaluation images.

Data (not committed) lives in `data/`:
- `train_labels.csv` (`id,filename,label`), `train_images/train_XXXXX.png`
- `test.csv` (`id,filename`), `test_images/test_XXXX.png`
- `sample_submission.csv` (`id,label`)

Rules: no pretrained weights, no external data, test images are never used for training.
