# Raw optical tactile RGB input

TouchWAM can read the FTP conversion's existing lossless optical RGB videos directly. It does not reconstruct a deform map, infer pressure from brightness, replicate grayscale into RGB, or change the original video files. RGB values divided by 255 represent optical color channels. They are not pressure or force labels.

The scene video and tactile videos remain separate. Scene color augmentation does not modify tactile RGB. Both tactile representations retain the same region identity and physical prediction times; a separate RGB encoder maps color images into the shared tactile stream.

## Reader configuration

```yaml
type: lerobot_robot_intensity  # Existing reader name; supports both representations.
source_kind: robot
source_family: ftp1
expected_modalities: [video_action_tactile]
require_publication: true
tactile_mode: required
tactile_representation: rgb
tactile_rgb_manifest: meta/tactile_rgb.json
tactile_rgb_map_size: 128
action_contract: meta/action_contract.json
action_mode: eef
normalize_mode: min-max
unify_action: false
```

The default `tactile_representation: intensity` retains the previous intensity reader and its 32×32 maps. `tactile_mode: off` remains valid for sources without tactile supervision. `required` never falls back to video-only data when the selected representation is missing. An RGB source does not substitute an existing intensity sidecar for its RGB manifest.

For an ego source with an equivalent LeRobot v3 release, `lerobot_video_intensity` can read RGB into VT samples. FTP sources must use the robot reader and a verified action contract. Raw robot state is not automatically interpreted as a command.

## Per-bucket RGB manifest

`meta/tactile_rgb.json` is a new metadata artifact prepared and verified by the data publisher. The following example shows one region; list all actual regions and every published episode. The image and validity feature names are the exact names already written by the FTP converter.

```json
{
  "schema": "tactile_rgb_v1",
  "codec": "tactile_rgb_v1",
  "representation": "optical_rgb",
  "normalization": "uint8_div_255",
  "time_alignment": "dataset_rows",
  "source_id": "ftp1/Unit",
  "dataset_info_sha256": "SHA256_OF_META_INFO_JSON",
  "status": "verified",
  "verification": {
    "evidence": ["Publisher's concrete sensor identity, image orientation and synchronization verification"]
  },
  "regions": [
    {
      "region_name": "right.gripper.jaw_a.surface",
      "sensor_name": "unit",
      "video_key": "observation.tactile_rgb.right_tactile_data_gripper.0",
      "validity_column": "observation.tactile_rgb.right_tactile_data_gripper.0.valid",
      "orientation": {
        "rotate_quadrants": 0,
        "flip_horizontal": false,
        "flip_vertical": false
      },
      "roi": {"type": "full"}
    }
  ],
  "episodes": {
    "0": {"layout_present": [true]},
    "1": {"layout_present": [true]}
  }
}
```

This is a schema example, not a verified assertion that a particular exported sensor is jaw A. The publisher must confirm that mapping from the actual source. A single-arm gripper uses the canonical right-arm A/B regions. Do not interpret `left_tactile`/`right_tactile` as two arms without source evidence. Region names must be unique; multiple native sensors for one public region must be composed into one image before this reader. `surface_id` is unsupported.

The manifest binds the source identity and `meta/info.json` fingerprint. The training data contract also records the manifest fingerprint, representation and output size, so changed mappings are visible to strict resume. Sensor names must also appear in the model's sensor registry; preflight and collation check this.

## Geometry and ROI

Orientation applies counterclockwise rotations of 0, 90, 180 or 270 degrees, followed by horizontal then vertical flips. It transforms the image and its ROI together. No data-dependent orientation or per-frame normalization is performed.

ROI choices are:

| Definition | Meaning |
| --- | --- |
| `{"type": "full"}` | All native image pixels are valid. |
| `{"type": "rectangle", "bounds": [x0,y0,x1,y1]}` | Half-open native-pixel rectangle; pixels outside it are unobserved. |
| `{"type": "mask", "path": "meta/roi_jaw_a.npy", "sha256": "..."}` | Nonempty boolean `[native_height,native_width]` mask, bound by SHA-256. |

ROIs mask the original canvas; they do not crop and stretch the selected patch. The oriented canvas is resized to the configured square size using area interpolation. Each output cell averages only supported source pixels. The corresponding one-channel output ROI marks cells with any support and applies to all three RGB channels. Unsupported pixels are zero-valued padding, never valid black observations. Arbitrary-shaped sensor surfaces may use the small `.npy` mask without reconverting videos.

## Time and validity contract

The first RGB reader version requires a regular, verified dataset clock and tactile video FPS equal to that clock. The FTP converter already encodes all retained features at the source's declared FPS. Irregular or independently sampled tactile streams need an explicit alignment step before this reader; it will not guess their synchronization.

The reader uses each tactile video's own episode `from_timestamp`, chunk index and file index, including when several episodes share one MP4. Video interval duration must agree with the episode row count. Row timestamps are validated and then sampled on the exact grid to avoid float32 boundary errors.

- History defaults to `[-0.1, -2/30, -1/30, 0]` seconds and uses causal sample-and-hold. Future packets never fill history.
- Future targets default to `[0.1, 0.2, ..., 0.8]` seconds. A target can use the closest measurement within tolerance only when that measurement is strictly after the current time and remains inside the episode or segment.
- Optional `history_max_age_s` and `future_tolerance_s` default to `1.5/native_fps` and `0.51/native_fps`.
- Every video mapping requires an explicit scalar boolean validity feature. Missing or null validity is false. Missing RGB packets produce zero tensors with false observation/target masks.
- An observed black RGB image remains valid. Pixel intensity never determines contact, packet availability or layout.
- Layout and future query masks describe which surfaces exist and should be predicted. They do not disappear merely because a future label is missing.

## Output and training grouping

For `H=4`, `F=8`, `N` active layout entries and the default RGB size:

| Sample field | Shape / value |
| --- | --- |
| `tactile_maps` | float32 `[4,N,3,128,128]`, color values in `[0,1]` |
| `future_tactile_maps` | float32 `[8,N,3,128,128]` |
| `tactile_valid` / `future_tactile_valid` | bool `[4,N]` / `[8,N]` |
| `tactile_spatial_valid` | bool `[N,1,128,128]` |
| `tactile_layout_present` | bool `[N]` |
| `tactile_query_mask` | bool `[8,N]` |
| `tactile_codec` | `tactile_rgb_v1` |
| Region and sensor identities | Explicit registered names, converted to IDs by collation |

Requested times and selected source times are emitted separately, as for intensity sidecars. A dataset exposes `tactile_codec` so global training batches can align both supervision mode and representation. One source's homogeneous buckets cannot mix RGB and intensity representations; declare separate sources if both representations are deliberately trained. The collator also rejects representation mixing in one microbatch.

## Publication and action gates

RGB needs verified media and sensor geometry, not deform calibration. Existing `source_publication_contract` checks remain enabled and unchanged: incomplete media, incomplete source populations, failed validation or unverified clocks still block training. Historical archival records may say that calibrated tactile supervision was unavailable; that does not manufacture the new RGB mapping. The RGB manifest supplies a separate verified representation contract.

For robots, `meta/action_contract.json` is still mandatory and must bind the current dataset metadata. It must provide verified canonical command or explicitly declared future-state supervision, masks, 80-slot mapping and compatible action/state normalization statistics. Neither the RGB reader nor this schema generates those action fields or upgrades an unverified action source. No real data publication has been changed by this code addition.
