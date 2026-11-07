# Third-party code

## cifar10-airbench (Keller Jordan), MIT License

Source: https://github.com/KellerJordan/cifar10-airbench (legacy/airbench96.py).
Adapted in this repository: the GPU crop and cutout functions in `src/data.py` and the flip / translate test-time augmentation in `src/predict.py`.

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

- Pre-activation ResNet: He et al. (2016), arXiv:1603.05027.
- SLA+SD: Lee, Hwang and Shin (2020), arXiv:1910.05872. Checked against https://github.com/hankook/SLA (`trainers.py`, `augmentations.py`).
- RegMixup: Pinto et al. (2022), arXiv:2206.14502.
- TrivialAugment: Müller and Hutter (2021), arXiv:2103.10158 (torchvision `TrivialAugmentWide`).
- Cutout: DeVries and Taylor (2017), arXiv:1708.04552.
- RandConv (random convolution augmentation, `--randconv-prob`): Xu et al. (2021), ICLR, arXiv:2007.13003.
- Squeeze-and-excitation (`--se-reduction`, ablation only): Hu et al. (2018), arXiv:1709.01507.

Ablations that are not in the final code (docs/LOGBOOK.md, Phase 15) used ideas from these papers:

- Progressive RandConv (stacked small kernels): Choi et al. (2023), CVPR, arXiv:2304.00424.
- Fourier amplitude mixing: Xu et al. (2021), FACT, CVPR, arXiv:2105.11120; Chen et al. (2021), APR, ICCV, arXiv:2108.08487.
