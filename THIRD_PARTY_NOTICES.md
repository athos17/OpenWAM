# Third-party model architecture attribution

`openwam/model/architectures/touchwam/heterogeneous.py` adapts the ViT front-end,
shared Transformer trunk and Fourier/CNN architecture from:

- FTP-1, Michael (Chengbo) Yuan and contributors, repository
  <https://github.com/michaelyuancb/ftp1-policy>, revision
  `89fa681d6c014cce28300946b7526db808e0b1c1`, Apache License 2.0.
  Full license: `licenses/FTP1-Apache-2.0.txt`.
- T3, Jialiang (Alan) Zhao, Yuxiang Ma, Lirui Wang and Edward H. Adelson,
  <https://github.com/alanzjl/t3>, MIT License.
  Full license and copyright notice: `licenses/T3-MIT.txt`.

This adaptation uses compatible tensor names and computation, independent sensor
front-ends and a shared image trunk. It adds masked spatial pooling, four-token
projections, source-specific normalization, strict checkpoint audits and TouchWAM
history/flow integration. FTP-1's CLS projection, functional-area embeddings,
action expert and policy are not copied into the TouchWAM model.

Pretrained assets are externally supplied from
`MJJJJ1064/ftp1_v0426_50kstep`, pinned revision
`d6e5b73e473d3e70fb5f53132a2b1c35b5031156`; the production configuration pins
each file SHA256. They are not vendored into this repository.
