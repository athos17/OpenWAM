#!/usr/bin/env python3
"""Read-only, bounded-memory QA of experimental Mini sidecars on BOS.

No original archive/video is decoded. Numeric sidecars are scanned in small
sequential chunks. This audits representation/provenance, not sensor calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openwam.tactile.registry import REGION_NAMES, REGISTRY_VERSION  # noqa: E402

SOURCES = ('Unit', 'Unit_Bimanual', 'VLA_touch')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1024**2), b''):
            result.update(value)
    return result.hexdigest()


def safe_path(root, value):
    result = (root / value).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f'Path outside artifact directory: {value}')
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_array(path, shape, dtype):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    require(value.shape == shape and value.dtype == np.dtype(dtype), f'Unexpected array shape/dtype: {path}')
    return value


def scan_maps(root, count, source_valid, identity_valid, reconstruct_valid, chunk_frames=16):
    """Full pixel checks with memory bounded by the chosen frame chunk."""
    signed = load_array(root / 'signed_height_60x80.npy', (count, 60, 80), 'float32')
    preview = load_array(root / 'positive_preview_32x32.npy', (count, 32, 32), 'float32')
    require(not np.any(reconstruct_valid & ~(source_valid & identity_valid)), 'Reconstruction valid without source/identity')
    if not source_valid[0] or not identity_valid[0]:
        require(not reconstruct_valid.any(), 'Reconstruction used an invalid reference candidate')
    lower, upper, preview_upper = None, None, None
    valid_pixels = saturated_pixels = 0
    representative = None
    max_peak = -1.
    max_preview = np.zeros((32, 32), np.float32)
    for begin in range(0, count, chunk_frames):
        end = min(begin + chunk_frames, count)
        valid = np.asarray(reconstruct_valid[begin:end])
        heights = np.asarray(signed[begin:end])
        intensities = np.asarray(preview[begin:end])
        require(np.isnan(heights[~valid]).all() and np.isnan(intensities[~valid]).all(), 'Invalid frames must be entirely NaN')
        if not valid.any():
            continue
        h = heights[valid]
        p = intensities[valid]
        require(np.isfinite(h).all() and np.isfinite(p).all(), 'Valid reconstruction contains NaN/Inf')
        require(p.min() >= 0 and p.max() <= 1, 'Fixed-scale preview outside [0,1]')
        local_min, local_max = float(h.min()), float(h.max())
        lower = local_min if lower is None else min(lower, local_min)
        upper = local_max if upper is None else max(upper, local_max)
        preview_upper = float(p.max()) if preview_upper is None else max(preview_upper, float(p.max()))
        valid_pixels += p.size
        saturated_pixels += int((p >= 1).sum())
        max_preview = np.maximum(max_preview, p.max(axis=0))
        peaks = p.max(axis=(1, 2))
        selected = int(peaks.argmax())
        if float(peaks[selected]) > max_peak:
            max_peak = float(peaks[selected])
            relative_frame = int(np.flatnonzero(valid)[selected])
            representative = {'frame_index': begin + relative_frame,
                              'preview': intensities[relative_frame].copy(),
                              'signed': heights[relative_frame].copy()}
    if reconstruct_valid[0]:
        require(not np.any(signed[0]) and not np.any(preview[0]), 'Candidate reference compared with itself should be zero')
    return {'signed_min': lower, 'signed_max': upper, 'preview_max': preview_upper,
            'preview_saturated_pixels': saturated_pixels,
            'preview_valid_pixels': valid_pixels,
            'preview_saturated_fraction': saturated_pixels / valid_pixels if valid_pixels else None,
            'representative_frame': representative['frame_index'] if representative else None}, representative, max_preview


def audit_surface(root, expected, plan, info, verify_hashes=True, full_archive=None, selected_archive=None):
    complete_path = root / 'complete.json'
    complete = json.loads(complete_path.read_text())
    count = int(complete['frames'])
    require(count > 0, 'Empty experimental surface')
    require(complete['status'] == 'complete_unverified_experiment', 'Surface is not a completed experiment')
    require(complete['training_enabled'] is False and complete['train_valid'] is False, 'Experiment wrongly enables training')
    require(complete['reference_confirmed_unloaded'] is False, 'Candidate reference mislabeled as unloaded')
    require(complete['reference_policy'] == 'episode-first-candidate' and complete['reference_frame_index'] == 0, 'Unexpected reference policy')
    require(complete['registry_version'] == REGISTRY_VERSION, 'Region vocabulary version differs')
    region = complete['region_name']
    require(region == expected['region_name'] and complete['region_id'] == REGION_NAMES.index(region), 'Region identity mismatch')
    require(complete['episode_index'] == expected['episode_index'], 'Episode identity mismatch')
    require(complete['fixed_preview_scale'] == plan['fixed_preview_scale'], 'Scale varies within an experiment')
    require(complete['normalization'] in ('clip(signed_height / fixed_scale, 0, 1), then area resize',
                                             'clip(area_resize(clip(signed_height / fixed_scale, 0, 1)), 0, 1)'), 'Unrecognized preview normalization')
    require(complete['options'] == plan['options'], 'Reconstruction options vary within experiment')
    require(complete['source_id'] == info['source_id'], 'Source identity mismatch')
    require(complete['height_units'] == 'relative_reconstruction_units_not_mm_or_force_or_pressure', 'Experimental height units changed')
    inputs = complete['inputs']
    signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    require(signature == complete['signature'], 'Surface signature does not match recorded inputs')
    artifact_paths = set()
    artifact_bytes = 0
    for record in complete['artifacts']:
        path = safe_path(root, record['path'])
        require(path.is_file() and path.stat().st_size == record['bytes'], f'Artifact missing/truncated: {path}')
        artifact_paths.add(record['path'])
        artifact_bytes += record['bytes']
        if verify_hashes:
            require(digest(path) == record['sha256'], f'Artifact SHA256 mismatch: {path}')
    mandatory = {'signed_height_60x80.npy', 'positive_preview_32x32.npy', 'source_valid.npy', 'identity_valid.npy',
                 'reconstruction_valid.npy', 'train_valid.npy', 'frame_index.parquet', 'reference_candidate.png',
                 'frame_diagnostics.jsonl', 'qa.png'}
    require(mandatory <= artifact_paths, 'Missing mandatory sidecar artifact')
    masks = {name: load_array(root / (name + '.npy'), (count,), 'bool')
             for name in ['source_valid', 'identity_valid', 'reconstruction_valid', 'train_valid']}
    require(not masks['train_valid'].any(), 'At least one experimental frame was enabled for training')
    for key in ['source_valid', 'identity_valid', 'reconstruction_valid']:
        require(int(masks[key].sum()) == complete[key + '_frames'], f'Mask count disagrees with metadata: {key}')
    metadata = pq.read_table(root / 'frame_index.parquet').to_pydict()
    require(len(metadata['frame_index']) == count, 'Frame metadata length mismatch')
    require(np.array_equal(metadata['frame_index'], np.arange(count)), 'Episode frame index is discontinuous')
    require(np.allclose(metadata['timestamp'], np.arange(count) / info['fps'], atol=1e-6, rtol=0), 'Unexpected episode clock')
    require(np.array_equal(metadata['episode_index'], np.full(count, complete['episode_index'])), 'Frame episode index disagrees')
    episode = inputs['episode']
    source_from = episode['source/from_frame']
    source_to = episode['source/to_frame']
    require(source_to - source_from == count, 'Source frame range length mismatch')
    require(np.array_equal(metadata['observation.ftp1.source_frame_index'], np.arange(source_from, source_to)), 'Original task frame identity lost')
    require(np.array_equal(metadata['observation.ftp1.source_episode_index'], np.full(count, episode['source/episode_index'])), 'Original episode identity lost')
    surface = inputs['surface']
    feature = 'observation.tactile_rgb.' + surface['raw_field'] + '.' + str(surface['raw_sensor_index'])
    require(np.array_equal(metadata[feature + '.valid'], masks['source_valid']), 'Source validity disagrees with preserved frame metadata')
    identity_fields = ['observation.ftp1.' + surface['raw_field'].replace('_data_', f'_{kind}_') for kind in ['area', 'sensor', 'type']]
    area_valid, sensor_valid, type_valid = [np.asarray(metadata[field + '.valid'], dtype=bool) for field in identity_fields]
    policy = complete.get('identity_policy', 'strict')
    require(policy == plan.get('identity_policy', 'strict'), 'Identity policy differs from experiment plan')
    derived = np.zeros(count, bool)
    if policy == 'declared_constant_area':
        proof = complete['identity_evidence']
        require(full_archive is not None and selected_archive is not None, 'Complete original source index proof required')
        require(proof == inputs['identity_evidence'], 'Identity evidence differs from signed input provenance')
        require(proof['full_source_archive_index_sha256'] == plan['full_source_archive_index']['sha256'], 'Full source proof hash differs')
        require(full_archive['archive_fingerprint'] == selected_archive['archive_fingerprint'] == proof['full_source_archive_fingerprint'], 'Source archive fingerprints differ')
        expected_mapping = {
            'Unit': {'right_tactile_data_gripper': 'right.gripper.jaw_a.surface'},
            'VLA_touch': {'right_tactile_data_gripper': 'right.gripper.jaw_a.surface'},
            'Unit_Bimanual': {'right_tactile_data_gripper': 'right.gripper.jaw_a.surface',
                             'left_tactile_data_gripper': 'left.gripper.jaw_a.surface'}}
        name = info['source_id'].removeprefix('ftp1/')
        require(expected_mapping.get(name, {}).get(surface['raw_field']) == region, 'Unregistered fallback region profile')
        require(surface['raw_sensor_index'] == 0 and surface['expected_ftp_area_id'] == 0 and surface['sensor'] == 'GelSightMini', 'Unregistered fallback sensor profile')
        field = 'data/' + surface['raw_field'].replace('_data_', '_area_')
        require(proof['array_field'] == field, 'Constant area evidence refers to another field')
        original = full_archive['arrays'][field]
        config = original['config']
        require(config == selected_archive['arrays'][field]['config'], 'Full and selected area configurations differ')
        require(original['chunks'] == {} and config.get('zarr_format') == 2 and len(config['shape']) == 2
                and config['shape'][1] == 1 and type(config['fill_value']) is int and config['fill_value'] == 0
                and np.issubdtype(np.dtype(config['dtype']), np.integer), 'Not a proven globally all-fill integer area array')
        derived = ~area_valid & sensor_valid & type_valid
        stored_derived = load_array(root / 'region_identity_from_declared_fill.npy', (count,), 'bool')
        require(np.array_equal(derived, stored_derived), 'Derived static region mask differs')
        require(int(derived.sum()) == complete['region_identity_from_declared_fill_frames'], 'Derived region count differs')
    else:
        require(policy == 'strict', 'Unknown identity policy')
        optional = root / 'region_identity_from_declared_fill.npy'
        if optional.exists():
            require(not load_array(optional, (count,), 'bool').any(), 'Strict identity policy used derived area IDs')
    recorded_identity = (area_valid | derived) & sensor_valid & type_valid
    require(np.array_equal(recorded_identity, masks['identity_valid']), 'Identity mask differs from source metadata and declared evidence')
    for i in np.flatnonzero(area_valid | derived):
        require(metadata[identity_fields[0]][i][surface['raw_sensor_index']] == surface['expected_ftp_area_id'], 'Source area ID differs')
    for i in np.flatnonzero(sensor_valid):
        require(metadata[identity_fields[1]][i] == 'GelSightMini', 'Source is not the declared Mini sensor')
    for i in np.flatnonzero(type_valid):
        require(metadata[identity_fields[2]][i] == 'image', 'Source is not an image sensor')
    ranges, representative, max_preview = scan_maps(root, count, masks['source_valid'], masks['identity_valid'], masks['reconstruction_valid'])
    diagnostic_count = 0
    with (root / 'frame_diagnostics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row['frame_index'] == diagnostic_count, 'Diagnostics frame indices are discontinuous')
            for key in masks:
                require(row[key] == bool(masks[key][diagnostic_count]), f'Diagnostics disagree with {key}')
            if 'region_identity_from_declared_fill' in row:
                require(row['region_identity_from_declared_fill'] == bool(derived[diagnostic_count]), 'Diagnostics disagree with derived identity')
            diagnostic_count += 1
    require(diagnostic_count == count, 'Diagnostics frame count differs')
    summary = {'source_id': complete['source_id'], 'episode_index': complete['episode_index'], 'region_name': region,
               'region_id': complete['region_id'], 'source_episode_index': episode['source/episode_index'],
               'source_frame_range': [source_from, source_to], 'frames': count,
               'source_valid': int(masks['source_valid'].sum()), 'identity_valid': int(masks['identity_valid'].sum()),
               'reconstruction_valid': int(masks['reconstruction_valid'].sum()), 'train_valid': 0,
               'identity_policy': policy, 'region_identity_from_declared_fill_frames': int(derived.sum()),
               'fixed_preview_scale': complete['fixed_preview_scale'], 'artifact_bytes': artifact_bytes,
               'artifact_sha256_verified': verify_hashes, 'complete_sha256': digest(complete_path), **ranges}
    tile = {'title': f'{complete["source_id"].removeprefix("ftp1/")} ep{complete["episode_index"]} {region}',
            'representative': representative, 'max_preview': max_preview, 'scale': complete['fixed_preview_scale']}
    return summary, tile


def montage(path, tiles):
    # At most 20 rows per image, each row 600x210. No full videos/images loaded.
    per_page = 20
    paths = []
    for page_start in range(0, len(tiles), per_page):
        selected = tiles[page_start:page_start + per_page]
        canvas = Image.new('RGB', (600, 80 + 210 * len(selected)), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), 'UNVERIFIED Mini geometry - no force/depth calibration; train_valid=false', fill='red')
        draw.text((8, 28), 'Peak-frame signed | Peak-frame 32x32 preview | Pixelwise temporal max', fill='black')
        draw.text((8, 46), 'One fixed scale. First frame is only a reference candidate.', fill='black')
        for i, tile in enumerate(selected):
            y = 80 + i * 210
            draw.text((5, y), tile['title'], fill='black')
            sample = tile['representative']
            if sample is None:
                draw.text((5, y + 50), 'No numerically valid reconstruction', fill='red')
                continue
            signed = np.clip(sample['signed'] / tile['scale'], -1, 1)
            colors = np.stack((np.clip(1 + signed, 0, 1), 1 - abs(signed), np.clip(1 - signed, 0, 1)), -1)
            images = [(255 * colors).astype(np.uint8), (255 * sample['preview']).astype(np.uint8),
                      (255 * tile['max_preview']).astype(np.uint8)]
            for j, image in enumerate(images):
                canvas.paste(Image.fromarray(image).convert('RGB').resize((180, 160), Image.Resampling.NEAREST), (5 + j * 200, y + 22))
            draw.text((5, y + 185), f'peak frame {sample["frame_index"]}; fixed scale {tile["scale"]:g} relative units', fill='black')
        target = path.with_name(path.stem + f'-{page_start // per_page:02d}' + path.suffix)
        canvas.save(target)
        paths.append(str(target))
    return paths


def audit_experiment(path, verify_hashes=True, bucket=None):
    record = json.loads(path.read_text())
    root = path.parent
    bucket = Path(bucket) if bucket is not None else root.parent.parent
    require(record['status'] == 'complete_unverified_experiment', 'Experiment is incomplete or unsupported')
    require(record['training_enabled'] is False and record['train_valid'] is False, 'Experiment enabled formal training')
    plan = record['plan']
    require(record['signature'] == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(), 'Experiment signature differs')
    info_path = bucket / 'meta' / 'info.json'
    info = json.loads(info_path.read_text())
    require(digest(info_path) == plan['source_info_sha256'], 'Source info changed after reconstruction')
    require(digest(bucket / 'meta' / 'source_definition.json') == plan['source_definition_sha256'], 'Source definition changed')
    require(info['source_id'] == 'ftp1/' + record['source'], 'Experiment source differs from v3 source')
    require(plan['reference_policy'] == 'episode-first-candidate', 'Unexpected baseline policy')
    require(np.isfinite(plan['fixed_preview_scale']) and plan['fixed_preview_scale'] > 0, 'Invalid fixed scale')
    require(not (bucket / 'tactile' / 'manifest.json').exists(), 'Formal tactile manifest unexpectedly present for this pilot')
    full_archive = selected_archive = None
    if plan.get('identity_policy', 'strict') == 'declared_constant_area':
        proof = plan['full_source_archive_index']
        proof_path = safe_path(root, proof['path'])
        require(proof_path.stat().st_size == proof['bytes'] and digest(proof_path) == proof['sha256'], 'Full source index proof hash differs')
        full_archive = json.loads(proof_path.read_text())
        selected_archive = json.loads((bucket / 'meta/archive_index.json').read_text())
        for evidence in plan['identity_evidence']:
            require(digest(bucket / 'meta/archive_index.json') == evidence['archive_index_sha256'], 'Selected v3 archive index changed')
    rows, tiles = [], []
    seen = set()
    for surface in record['surfaces']:
        key = (surface['episode_index'], surface['region_name'])
        require(key not in seen, 'Duplicate episode/public region')
        seen.add(key)
        summary, tile = audit_surface(safe_path(root, surface['complete']).parent, surface, plan, info, verify_hashes, full_archive, selected_archive)
        rows.append(summary)
        tiles.append(tile)
    require(len(rows) == record['episode_surface_count'], 'Surface count differs from experiment summary')
    require(sum(row['frames'] for row in rows) == record['surface_frames'], 'Surface frame count differs')
    require(sum(row['reconstruction_valid'] for row in rows) == record['reconstruction_valid_surface_frames'], 'Valid reconstruction count differs')
    unique = {}
    for row in rows:
        ep = row['episode_index']
        require(ep not in unique or unique[ep] == row['frames'], 'Surface episode lengths differ')
        unique[ep] = row['frames']
    require(len(unique) == record['episode_count'] and sum(unique.values()) == record['frames'], 'Unique episode/frame totals differ')
    return {'source': record['source'], 'experiment': str(path), 'episodes': len(unique), 'frames': sum(unique.values()),
            'surface_frames': sum(row['frames'] for row in rows), 'all_train_valid_false': True,
            'fixed_preview_scale': plan['fixed_preview_scale'], 'options': plan['options'], 'surfaces': rows,
            'validated': True, 'validation_scope': 'format_masks_provenance_and_numerics_only_not_sensor_calibration'}, tiles


def run(root, output, sources=SOURCES, verify_hashes=True, variant='mini_v2'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports, errors, tiles = [], [], []
    for source in sources:
        matches = sorted((Path(root) / source).glob(f'**/experimental_deform/{variant}/experiment.json'))
        if not matches:
            errors.append({'source': source, 'error': 'No completed experimental manifest found yet'})
        for path in matches:
            try:
                report, images = audit_experiment(path, verify_hashes)
                reports.append(report)
                tiles.extend(images)
            except (ValueError, KeyError, OSError, IndexError) as error:
                errors.append({'source': source, 'path': str(path), 'error': str(error)})
    figures = montage(output / 'deform-audit.png', tiles)
    result = {'status': 'passed_representation_audit' if not errors else 'incomplete_or_failed',
              'sensor_calibration_validated': False, 'training_enabled': False, 'source_root': str(root),
              'artifact_sha256_verified': verify_hashes, 'experiments': reports, 'errors': errors, 'figures': figures,
              'notes': ['Only sidecars and small metadata are read; original archive/video not decoded.',
                        '60x80 signed and 32x32 preview are separately resized from native height.',
                        'Exact normalization cannot be recomputed from the stored 60x80 map; config and provenance are checked.',
                        'Reference candidate is not confirmed unloaded; no mm/N/Pa calibration.',
                        'Numerical reconstruction validity is not training validity.']}
    (output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'experiments'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='BOS mount containing Unit/Unit_Bimanual/VLA_touch')
    parser.add_argument('--output', required=True, help='Small audit report directory outside the input dataset')
    parser.add_argument('--sources', nargs='+', choices=SOURCES, default=list(SOURCES))
    parser.add_argument('--variant', choices=['mini_v1', 'mini_v2'], default='mini_v2')
    parser.add_argument('--skip-artifact-hashes', action='store_true', help='Faster numeric-only recheck; normal audit verifies all sidecar hashes')
    args = parser.parse_args()
    report = run(args.root, args.output, args.sources, not args.skip_artifact_hashes, args.variant)
    sys.exit(0 if report['status'] == 'passed_representation_audit' else 1)
