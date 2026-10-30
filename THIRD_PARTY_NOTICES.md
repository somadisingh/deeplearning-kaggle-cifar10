# Third-party code

## cifar10-airbench (Keller Jordan), MIT License

Source: https://github.com/KellerJordan/cifar10-airbench (legacy/airbench96.py).
Adapted in this repository: the AirResNet blocks, dirac conv init, BatchNorm with fixed scale and the patch-whitening initialization in `src/models.py`; the GPU crop / cutout functions, the decoupled SGD hyperparameters, the Lookahead EMA and the flip / translate test-time augmentation in `src/train.py`.

```
Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Methods re-implemented from papers (no code copied)

- SLA+SD: Lee, Hwang and Shin (2020), arXiv:1910.05872. Checked against https://github.com/hankook/SLA (`trainers.py`, `augmentations.py`).
- Auxiliary rotation head: Hendrycks et al. (2019), arXiv:1906.12340. Checked against https://github.com/hendrycks/ss-ood.
- SAM: Foret et al. (2021), arXiv:2010.01412.
- PAPA: Jolicoeur-Martineau et al. (2024), arXiv:2304.03094 (Algorithm 1).
- Shake-Shake: Gastaldi (2017), arXiv:1705.07485.
- BlurPool anti-aliasing: Zhang (2019), arXiv:1904.11486.
- Pre-activation ResNet: He et al. (2016), arXiv:1603.05027.
- TrivialAugment: Müller and Hutter (2021), arXiv:2103.10158 (via `torchvision.transforms.v2.TrivialAugmentWide`).
