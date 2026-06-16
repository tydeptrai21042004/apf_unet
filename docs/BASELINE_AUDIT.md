# Baseline implementation audit

This repository separates two claims:

- **paper-aligned architecture**: the implementation contains the named backbone, modules, prediction heads, and supervision graph described by the paper and official code;
- **byte-identical official reproduction**: not claimed, because all models are adapted to one common training/evaluation interface and emit raw logits.

## Audited baselines

| Model | Paper-aligned contract enforced | Official source | Notes |
|---|---|---|---|
| U-Net | encoder-decoder with four skip paths | Ronneberger et al. | padded same-size benchmark adaptation |
| Attention U-Net | additive attention gates on decoder skips | Oktay et al. | common PyTorch adaptation |
| U-Net++ | nested dense skip topology and optional deep supervision | Zhou et al. | raw-logit output adaptation |
| ResUNet++ | residual units, SE, ASPP bridge, attention decoder | Jha et al. | unified channels/output contract |
| PraNet | Res2Net-50, three RFBs, PPD aggregation, RA4→RA3→RA2, four maps | DengPingFan/PraNet | official-style backbone and head graph |
| ACSNet | **ResNet-34**, LCA, GCM, ASM, five prediction levels | ReaFly/ACSNet | corrected from the previous incorrect Res2Net encoder |
| HarDNet-MSEG | HarDNet-68, RFB projections, cascaded aggregation decoder | james128333/HarDNet-MSEG | raw-logit adaptation |
| Polyp-PVT | PVTv2-B2, CFM, CIM, SAM, coarse and refined maps | DengPingFan/Polyp-PVT | official-style backbone/modules |
| CaraNet | Res2Net-50, CFP dilation context, axial reverse attention chain | AngeLouCN/CaraNet | official-style backbone/modules |
| CFA-Net | boundary prediction, two streams, CFF, BAM, boundary supervision | taozh2017/CFANet | unified output dictionary |
| HSNet | CNN/PVT dual encoders, CSA, HSC, MSP | baiboat/HSNet | unified output dictionary |
| CSCA U-Net | six stages, DSE bottleneck, CSCA, CLFF, deep supervision | xiaolanshu/CSCA-U-Net | efficient attention is only for smoke tests |

## Reproducibility rule

Use `configs/official_faithful/` for paper-aligned backbone/module contracts and `configs/fair/` for controlled architecture comparisons. Passing tests proves structural and functional alignment, not numerical equivalence to an authors' checkpoint.
