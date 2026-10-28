#!/usr/bin/env python3
"""Stream FTP-1 Mini videos into explicitly UNVERIFIED geometry sidecars.

This experiment is deliberately outside ``tactile/manifest.json``. It never
publishes training labels. Episode-first references are candidates, NOT known
unloaded images. Signed differences are relative reconstruction units, not mm,
newtons or pressure. Only native 240x320 Mini sources are currently supported.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path
import sys

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from openwam.tactile.registry import REGION_NAMES, REGISTRY_VERSION  # noqa: E402

try:
    from .depth_plugins import gelsight_mini as mini
except ImportError:
    from depth_plugins import gelsight_mini as mini

VERSION = 'touchwam.ftp1.experimental_mini_deform.v1'
DECLARED_AREA_VERSION = 'touchwam.ftp1.experimental_mini_deform.v2'
SUPPORTED = {'Unit', 'Unit_Bimanual', 'VLA_touch'}
CHECKPOINT_SHA256 = '091bb7ee7ac259d8f77e80ca61c0370c09af6a4659bc9a4222474d4eca0bd65e'
UNSUPPORTED = {
    'RDP': 'Mini input is 224x224 with unverified crop/FOV; MCTac has a different optical model.',
    'RDP_Bimanual': 'Mini input is 224x224 with unverified crop/FOV; MCTac has a different optical model.',
    'FreeTacMan': 'Modified MCTac optics; no matching normal/depth calibration verified.',
    'exUMI': 'Processed 9DTact channels need matching LUT, reference, crop and rotation identity.',
    'ViTaMIn': 'AllTact Fin Ray images mix whole-finger bending and local contact; need its own geometry model.',
    'VisuoTactile_D-WHEEL': 'OpenLoongVTouch optics have no verified matching reconstruction calibration.',
    'VisuoTactile_QINGLOONG': 'OpenLoongVTouch optics have no verified matching reconstruction calibration.',
    'RH20TCfg7Tactile': 'uSkin is a spatial vector array, not an RGB camera; requires normal-axis and baseline calibration.',
    'RH20TCfg5Franka': 'Franka torque is not a spatial optical measurement.',
    'RH20TCfg6ATIAxia': 'ATI wrench cannot uniquely determine a contact pressure map.',
    'MotionTrans': 'Joint force signals are not spatial optical measurements.',
}


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def relative_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Artifact escapes its root: {value}')
    return path


def load_weights(root):
    root = Path(root)
    record = json.loads((root / 'provenance.json').read_text())
    if record['source_checkpoint_sha256'] != CHECKPOINT_SHA256:
        raise ValueError('Unexpected public Mini checkpoint identity')
    weights = {}
    for key, shape in mini.WEIGHT_SHAPES.items():
        # Paths in the export provenance describe its original host. Resolve
        # portable numeric arrays by their expected basename and verify hashes.
        path = root / (key + '.npy')
        if sha256(path) != record['arrays'][key]['sha256']:
            raise ValueError(f'Numeric calibration hash mismatch: {key}')
        value = np.load(path, allow_pickle=False)
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'Invalid numeric calibration tensor: {key}')
        weights[key] = value
    return weights, record


def episode_rows(bucket):
    rows = []
    for path in sorted((bucket / 'meta' / 'episodes').glob('chunk-*/*.parquet')):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: row['episode_index'])
    if len({row['episode_index'] for row in rows}) != len(rows):
        raise ValueError('Duplicate episode metadata')
    return rows


def declared_area_evidence(bucket, info, surface, full_archive_index):
    """Narrow opt-in for a registered single-pad region with Zarr fill-only area.

    A missing *measurement* is never repaired this way. This only resolves the
    static region identity from a source archive's explicit constant area field.
    """
    source = info['source_id'].removeprefix('ftp1/')
    allowed = {
        'Unit': {'right_tactile_data_gripper': 'right.gripper.jaw_a.surface'},
        'VLA_touch': {'right_tactile_data_gripper': 'right.gripper.jaw_a.surface'},
        'Unit_Bimanual': {'right_tactile_data_gripper': 'right.gripper.jaw_a.surface',
                         'left_tactile_data_gripper': 'left.gripper.jaw_a.surface'},
    }
    expected_region = allowed.get(source, {}).get(surface['raw_field'])
    if (surface['canonical_region'] != expected_region or surface['sensor'] != 'GelSightMini'
            or surface['raw_sensor_index'] != 0 or surface['expected_ftp_area_id'] != 0):
        raise ValueError('Declared constant area fallback is not registered for this exact source/surface profile')
    path = bucket / 'meta' / 'archive_index.json'
    selected_archive = json.loads(path.read_text())
    if full_archive_index is None:
        raise ValueError('A complete source archive index is required; a selected sparse index is not proof of an all-fill field')
    full_path = Path(full_archive_index)
    archive = json.loads(full_path.read_text())
    if archive['archive_fingerprint'] != selected_archive['archive_fingerprint']:
        raise ValueError('Complete source index fingerprint does not match the v3 source archive')
    field = 'data/' + surface['raw_field'].replace('_data_', '_area_')
    array = archive['arrays'][field]
    config = array['config']
    if config != selected_archive['arrays'][field]['config']:
        raise ValueError('Complete source index array configuration does not match v3 metadata')
    shape = config['shape']
    fill = config.get('fill_value')
    if (config.get('zarr_format') != 2 or len(shape) != 2 or shape[1] != 1
            or array.get('chunks') != {} or type(fill) is not int or fill != 0
            or not np.issubdtype(np.dtype(config['dtype']), np.integer)):
        raise ValueError('Declared constant area fallback requires a single-pad integer Zarr v2 field with fill_value=0 and no stored chunks')
    return {'policy': 'declared_constant_area', 'archive_index_path': 'meta/archive_index.json',
            'archive_index_sha256': sha256(path),
            'full_source_archive_index_path': 'full_source_archive_index.json',
            'full_source_archive_index_sha256': sha256(full_path),
            'full_source_archive_fingerprint': archive['archive_fingerprint'],
            'array_field': field,
            'original_array_shape': shape, 'original_dtype': config['dtype'],
            'zarr_format': 2, 'declared_fill_value': 0, 'stored_chunks': 0,
            'region_name': expected_region, 'original_area_valid_preserved': True,
            'scope': 'static_region_identity_only_never_tactile_measurements'}


def episode_metadata(bucket, info, episode, feature, surface, identity_policy='strict', full_archive_index=None):
    path = relative_path(bucket, info['data_path'].format(
        chunk_index=episode['data/chunk_index'], file_index=episode['data/file_index']))
    schema = pq.read_schema(path)
    columns = ['index', 'episode_index', 'frame_index', 'timestamp', feature + '.valid']
    columns += [key for key in schema.names if key.startswith('source/') or key.startswith('observation.ftp1.source_')]
    identity_fields = ['observation.ftp1.' + surface['raw_field'].replace('_data_', f'_{kind}_') for kind in ('area', 'sensor', 'type')]
    columns += [key for field in identity_fields for key in (field, field + '.valid')]
    # source/ fields are retained without assuming their physical meaning.
    table = pq.read_table(path, columns=columns,
                          filters=[('episode_index', '=', episode['episode_index'])])
    if table.num_rows != episode['length']:
        raise ValueError('Episode metadata length mismatch')
    result = table.to_pydict()
    frame_index = np.asarray(result['frame_index'])
    timestamp = np.asarray(result['timestamp'], dtype=np.float64)
    if not np.array_equal(frame_index, np.arange(episode['length'])):
        raise ValueError('Expected contiguous episode-local frame indices')
    if not np.allclose(timestamp, frame_index / info['fps'], atol=1e-6, rtol=0):
        raise ValueError('Irregular/mismatched video clock')
    valid = result[feature + '.valid']
    if any(type(value) is not bool for value in valid):
        raise ValueError('Source frame validity must be an explicit non-null bool')
    area_valid, sensor_valid, type_valid = [np.asarray(result[field + '.valid'], dtype=bool) for field in identity_fields]
    evidence = None
    derived_region = np.zeros(episode['length'], dtype=bool)
    if identity_policy == 'declared_constant_area':
        evidence = declared_area_evidence(bucket, info, surface, full_archive_index)
        derived_region = ~area_valid & sensor_valid & type_valid
    elif identity_policy != 'strict':
        raise ValueError('Unsupported identity policy')
    # Observed metadata must be consistent even if another identity field is
    # missing; a fallback may never conceal an observed contradiction.
    for i in np.flatnonzero(area_valid | derived_region):
        area = np.asarray(result[identity_fields[0]][i]).reshape(-1)
        sensor_index = surface['raw_sensor_index']
        if sensor_index >= len(area) or area[sensor_index] != surface['expected_ftp_area_id']:
            raise ValueError('Recorded tactile area disagrees with the public region mapping')
    for i in np.flatnonzero(sensor_valid):
        if result[identity_fields[1]][i] != surface['sensor']:
            raise ValueError('Recorded sensor identity/type does not match the Mini optical profile')
    for i in np.flatnonzero(type_valid):
        if result[identity_fields[2]][i] != 'image':
            raise ValueError('Recorded sensor identity/type does not match the Mini optical profile')
    identity_valid = (area_valid | derived_region) & sensor_valid & type_valid
    return result, path, identity_valid, derived_region, evidence


def video_frames(path, start, length, fps):
    """Decode one episode sequentially after seeking to a preceding keyframe."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        container.seek(max(0, int(start / float(stream.time_base))), stream=stream,
                       backward=True, any_frame=False)
        emitted = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Video frame has no timestamp')
            clock = float(frame.pts * stream.time_base)
            if clock < start - .25 / fps:
                continue
            expected = start + emitted / fps
            if abs(clock - expected) > .25 / fps:
                raise ValueError(f'Video clock mismatch: {clock} versus {expected}')
            yield frame.to_ndarray(format='rgb24')
            emitted += 1
            if emitted == length:
                return
        raise ValueError(f'Video ended after {emitted}/{length} episode frames')


def low_resolution_maps(current, reference, scale, height_size=(60, 80)):
    signed = (current - reference).astype(np.float32)
    low = cv2.resize(signed, (height_size[1], height_size[0]), interpolation=cv2.INTER_AREA)
    # Clip before area resize, matching the declared fixed-scale display codec.
    positive = np.clip(signed / scale, 0, 1)
    small = cv2.resize(positive, (32, 32), interpolation=cv2.INTER_AREA)
    # INTER_AREA can overshoot by one float32 ULP for saturated patches.
    # Preserve every in-range value and enforce the declared output boundary.
    np.clip(small, 0, 1, out=small)
    return low, small, signed


def qa_panel(path, samples, reference, scale, title):
    tile_w, tile_h = 320, 240
    canvas = Image.new('RGB', (tile_w * 4, 100 + 280 * len(samples)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill='black')
    draw.text((8, 27), 'UNVERIFIED: reference candidate, not confirmed unloaded. Not pressure.', fill='red')
    draw.text((8, 46), f'One fixed scale = {scale:g} reconstruction units. Training disabled.', fill='black')
    for col, label in enumerate(['Stored tactile RGB', 'Reference candidate', 'Signed height difference', '32x32 positive preview']):
        draw.text((8 + col * tile_w, 75), label, fill='black')
    for row, sample in enumerate(samples):
        offset = 100 + row * 280
        images = [sample['image'], reference]
        if sample['signed'] is None:
            images += [np.full((tile_h, tile_w, 3), 128, np.uint8)] * 2
        else:
            signed = np.clip(sample['signed'] / scale, -1, 1)
            color = np.stack((np.clip(1 + signed, 0, 1), 1 - abs(signed), np.clip(1 - signed, 0, 1)), -1)
            color = (255 * color).astype(np.uint8)
            gray = np.repeat((sample['small'][..., None] * 255).astype(np.uint8), 3, axis=2)
            images += [color, gray]
        for col, value in enumerate(images):
            canvas.paste(Image.fromarray(value).resize((tile_w, tile_h), Image.Resampling.NEAREST), (col * tile_w, offset))
        draw.text((8, offset + 245), f'frame {sample["frame"]}: {sample["status"]}', fill='black')
    canvas.save(path)


def artifact_record(path, root):
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def complete_unchanged(output, signature):
    path = output / 'complete.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result['signature'] != signature:
        raise ValueError(f'Existing experiment differs; select a new output directory: {output}')
    for record in result['artifacts']:
        path = relative_path(output, record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Completed experimental artifact failed its hash: {path}')
    return result


def process_surface(bucket, info, episode, surface, output, weights, options, scale, signature, identity_policy='strict', full_archive_index=None):
    index = int(episode['episode_index'])
    feature = 'observation.tactile_rgb.' + surface['raw_field'] + '.' + str(surface['raw_sensor_index'])
    if feature not in info['features'] or info['features'][feature]['shape'] != [240, 320, 3]:
        raise ValueError('Mini pilot requires the native declared 240x320 tactile video')
    region = surface['canonical_region']
    target = output / f'episode-{index:06d}' / region
    target.mkdir(parents=True, exist_ok=True)
    metadata, data_file, identity_valid, derived_region, identity_evidence = episode_metadata(
        bucket, info, episode, feature, surface, identity_policy, full_archive_index)
    video = relative_path(bucket, info['video_path'].format(video_key=feature,
        chunk_index=episode[f'videos/{feature}/chunk_index'], file_index=episode[f'videos/{feature}/file_index']))
    inputs = {'episode': episode, 'surface': surface, 'source_info_sha256': sha256(bucket / 'meta' / 'info.json'),
              'video': {'path': video.relative_to(bucket).as_posix(), 'sha256': sha256(video)},
              'data': {'path': data_file.relative_to(bucket).as_posix(), 'sha256': sha256(data_file)},
              'experiment_signature': signature, 'identity_policy': identity_policy, 'identity_evidence': identity_evidence}
    own_signature = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    existing = complete_unchanged(target, own_signature)
    if existing is not None:
        return existing
    count = episode['length']
    source_valid = np.asarray(metadata[feature + '.valid'], dtype=bool)
    reconstructed = np.zeros(count, dtype=bool)
    signed_maps = np.lib.format.open_memmap(target / 'signed_height_60x80.npy', mode='w+', dtype='float32', shape=(count, 60, 80))
    previews = np.lib.format.open_memmap(target / 'positive_preview_32x32.npy', mode='w+', dtype='float32', shape=(count, 32, 32))
    signed_maps[:] = np.nan
    previews[:] = np.nan
    candidates = set(np.linspace(0, count - 1, min(count, 5), dtype=int).tolist())
    samples, errors = [], []
    start = episode[f'videos/{feature}/from_timestamp']
    reference = None
    reference_surface = None
    valid_reference = False
    with (target / 'frame_diagnostics.jsonl').open('w') as diagnostics:
        for frame_index, image in enumerate(video_frames(video, start, count, info['fps'])):
            if image.shape != (240, 320, 3):
                raise ValueError('Decoded Mini image shape differs from native optical geometry')
            if frame_index == 0:
                reference = image.copy()
                Image.fromarray(reference).save(target / 'reference_candidate.png')
                if source_valid[0] and identity_valid[0]:
                    try:
                        reference_surface, reference_stats = mini.surface(reference, weights, options)
                        valid_reference = True
                    except ValueError as error:
                        errors.append({'frame': 0, 'stage': 'reference', 'error': str(error)})
            stats = {'frame_index': frame_index, 'source_valid': bool(source_valid[frame_index]),
                     'reference_source_valid': bool(source_valid[0]), 'identity_valid': bool(identity_valid[frame_index]),
                     'region_identity_from_declared_fill': bool(derived_region[frame_index]),
                     'reconstruction_valid': False, 'train_valid': False}
            signed = small = None
            if not source_valid[frame_index]:
                stats['status'] = 'missing_source_frame'
            elif not identity_valid[frame_index]:
                stats['status'] = 'missing_sensor_identity'
            elif not valid_reference:
                stats['status'] = 'reference_candidate_missing_or_numerically_invalid'
            else:
                try:
                    current, current_stats = (reference_surface, reference_stats) if frame_index == 0 else mini.surface(image, weights, options)
                    low, small, signed = low_resolution_maps(current, reference_surface, scale)
                    signed_maps[frame_index] = low
                    previews[frame_index] = small
                    reconstructed[frame_index] = True
                    stats.update(status='unverified_relative_geometry', reconstruction_valid=True,
                                 signed_min=float(signed.min()), signed_max=float(signed.max()),
                                 positive_saturation_fraction=float((signed >= scale).mean()), **current_stats)
                except ValueError as error:
                    stats.update(status='numerical_reconstruction_rejected', error=str(error))
                    errors.append({'frame': frame_index, 'stage': 'current', 'error': str(error)})
            diagnostics.write(json.dumps(stats, allow_nan=False) + '\n')
            if frame_index in candidates:
                samples.append({'frame': frame_index, 'image': image, 'signed': signed, 'small': small, 'status': stats['status']})
    signed_maps.flush()
    previews.flush()
    del signed_maps, previews
    np.save(target / 'source_valid.npy', source_valid, allow_pickle=False)
    np.save(target / 'identity_valid.npy', identity_valid, allow_pickle=False)
    np.save(target / 'region_identity_from_declared_fill.npy', derived_region, allow_pickle=False)
    np.save(target / 'reconstruction_valid.npy', reconstructed, allow_pickle=False)
    np.save(target / 'train_valid.npy', np.zeros(count, dtype=bool), allow_pickle=False)
    pq.write_table(pa.Table.from_pydict(metadata), target / 'frame_index.parquet', compression='zstd')
    qa_panel(target / 'qa.png', samples, reference, scale, f'{info["source_id"]} | {region} | episode {index}')
    record = {'schema_version': DECLARED_AREA_VERSION if identity_policy == 'declared_constant_area' else VERSION, 'signature': own_signature, 'status': 'complete_unverified_experiment',
              'training_enabled': False, 'train_valid': False, 'source_id': info['source_id'],
              'region_name': region, 'region_id': REGION_NAMES.index(region), 'registry_version': REGISTRY_VERSION,
              'episode_index': index, 'source_episode_index': episode.get('source/episode_index'),
              'source_group': episode.get('source/group'), 'frames': count, 'source_valid_frames': int(source_valid.sum()),
              'identity_valid_frames': int(identity_valid.sum()),
              'region_identity_from_declared_fill_frames': int(derived_region.sum()),
              'identity_policy': identity_policy, 'identity_evidence': identity_evidence,
              'reconstruction_valid_frames': int(reconstructed.sum()), 'reference_policy': 'episode-first-candidate',
              'reference_frame_index': 0, 'reference_confirmed_unloaded': False,
              'height_units': 'relative_reconstruction_units_not_mm_or_force_or_pressure',
              'fixed_preview_scale': scale, 'normalization': 'clip(area_resize(clip(signed_height / fixed_scale, 0, 1)), 0, 1)',
              'signed_height_shape': [count, 60, 80], 'positive_preview_shape': [count, 32, 32],
              'missing_values': 'NaN with false source/reconstruction mask; never measured zero',
              'signed_height_anchor': 'difference of two independently mean-zero Neumann reconstructions',
              'reference_limitation': 'Pre-existing static contact is subtracted; sign, RGB/XY, FOV and metric scale remain unverified.',
              'inputs': inputs, 'options': options, 'numerical_errors': errors,
              'artifacts': [artifact_record(path, target) for path in sorted(target.iterdir()) if path.is_file() and path.name not in ('complete.json', 'complete.json.tmp')]}
    atomic_json(target / 'complete.json', record)
    print(json.dumps({'episode': index, 'region': region, 'frames': count, 'reconstruction_valid': int(reconstructed.sum()), 'train_valid': False}), flush=True)
    return record


def run(args):
    bucket = Path(args.bucket).resolve()
    identity_policy = getattr(args, 'identity_policy', 'strict')
    full_archive_index = getattr(args, 'source_archive_index', None)
    variant = 'mini_v2' if identity_policy == 'declared_constant_area' else 'mini_v1'
    version = DECLARED_AREA_VERSION if identity_policy == 'declared_constant_area' else VERSION
    output = Path(args.output).resolve() if args.output else bucket / 'experimental_deform' / variant
    if output == bucket or (output.is_relative_to(bucket) and 'tactile' in output.relative_to(bucket).parts):
        raise ValueError('Experimental output must not overwrite the bucket or formal tactile cache')
    info_path = bucket / 'meta' / 'info.json'
    definition_path = bucket / 'meta' / 'source_definition.json'
    if not (bucket / 'meta' / 'conversion_complete.json').is_file():
        raise ValueError('A completed v3 conversion marker is required')
    info = json.loads(info_path.read_text())
    if info.get('codebase_version') != 'v3.0' or not np.isfinite(info['fps']) or info['fps'] <= 0:
        raise ValueError('Expected v3.0 metadata and a positive nominal video clock')
    definition = json.loads(definition_path.read_text())
    source = info['source_id'].removeprefix('ftp1/')
    if source not in SUPPORTED:
        record = {'schema_version': VERSION, 'source': source, 'status': 'unsupported_optical_calibration',
                  'training_enabled': False, 'train_valid': False,
                  'reason': UNSUPPORTED.get(source, 'No verified Mini-compatible native sensor definition'),
                  'source_info_sha256': sha256(info_path)}
        atomic_json(output / 'experiment.json', record)
        return record
    weights, provenance = load_weights(args.weights)
    regions = definition['tactile_region_definition']
    if regions['registry_version'] != REGISTRY_VERSION:
        raise ValueError('Region registry version mismatch')
    surfaces = regions['surfaces']
    if not surfaces or any(surface['sensor'] != 'GelSightMini' for surface in surfaces):
        raise ValueError('This pilot cannot apply Mini weights to another sensor')
    names = [surface['canonical_region'] for surface in surfaces]
    if len(names) != len(set(names)) or set(names) - set(REGION_NAMES):
        raise ValueError('Unique canonical regions are required')
    options = {'input_range': '0_255', 'image_shape': [240, 320], 'model_channel_order': args.channel_order,
               'xy_mode': args.xy_mode, 'zero_anchor': 'mean', 'indentation_sign': 1, 'output_scale': 1,
               'marker_gray_threshold': 70, 'max_unreliable_fraction': .5}
    plan = {'schema_version': version, 'identity_policy': identity_policy, 'source': source, 'source_info_sha256': sha256(info_path),
            'source_definition_sha256': sha256(definition_path), 'backend_sha256': sha256(mini.__file__),
            'runner_sha256': sha256(__file__), 'checkpoint_sha256': CHECKPOINT_SHA256,
            'options': options, 'fixed_preview_scale': args.preview_scale,
            'preview_post_resize_boundary_clamp': '[0,1]; float32 interpolation roundoff only',
            'reference_policy': args.reference_policy, 'max_episodes': args.episodes}
    if identity_policy == 'declared_constant_area':
        plan['identity_evidence'] = [declared_area_evidence(bucket, info, surface, full_archive_index) for surface in surfaces]
        plan['full_source_archive_index'] = {'path': 'full_source_archive_index.json',
                                             'sha256': sha256(full_archive_index),
                                             'bytes': Path(full_archive_index).stat().st_size}
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    old_path = output / 'plan.json'
    if old_path.exists() and json.loads(old_path.read_text()) != plan:
        raise ValueError('Existing experiment configuration changed; use a new output directory')
    atomic_json(old_path, plan)
    if identity_policy == 'declared_constant_area':
        proof = output / 'full_source_archive_index.json'
        if Path(full_archive_index).resolve() != proof.resolve():
            shutil.copyfile(full_archive_index, proof.with_suffix('.json.tmp'))
            proof.with_suffix('.json.tmp').replace(proof)
        if sha256(proof) != plan['full_source_archive_index']['sha256']:
            raise ValueError('Source archive proof changed while copying')
    atomic_json(output / 'weight_provenance.json', provenance)
    episodes = episode_rows(bucket)[:args.episodes]
    if not episodes:
        raise ValueError('No episodes in the completed v3 bucket')
    records = []
    for episode in episodes:
        for surface in surfaces:
            records.append(process_surface(bucket, info, episode, surface, output, weights, options, args.preview_scale, signature, identity_policy, full_archive_index))
    summary = {'schema_version': version, 'identity_policy': identity_policy, 'status': 'complete_unverified_experiment', 'source': source,
               'training_enabled': False, 'train_valid': False, 'episode_count': len(episodes),
               'episode_surface_count': len(records), 'frames': sum(ep['length'] for ep in episodes),
               'surface_frames': sum(record['frames'] for record in records),
               'reconstruction_valid_surface_frames': sum(record['reconstruction_valid_frames'] for record in records),
               'signature': signature, 'plan': plan,
               'surfaces': [{'episode_index': record['episode_index'], 'region_name': record['region_name'],
                             'complete': f'episode-{record["episode_index"]:06d}/{record["region_name"]}/complete.json'} for record in records],
               'limitations': ['Reference is not confirmed unloaded', 'No optical calibration validation',
                               'No sign/metric-depth/force calibration', 'Original RGB stored unchanged',
                               'No formal tactile manifest emitted; all train_valid are false']}
    atomic_json(output / 'experiment.json', summary)
    lines = ['<!doctype html><meta charset="utf-8"><title>Experimental Mini reconstruction</title>',
             '<h1>UNVERIFIED Mini relative geometry</h1><p>Reference candidates are not confirmed unloaded. Not pressure. Training disabled.</p>']
    for record in summary['surfaces']:
        image_path = str(Path(record['complete']).parent / 'qa.png')
        lines.append(f'<h2>{html.escape(record["region_name"])} / episode {record["episode_index"]}</h2><img width="100%" src="{html.escape(image_path)}">')
    (output / 'index.html').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bucket', required=True, help='Completed FTP LeRobot v3 task bucket')
    parser.add_argument('--output', help='Default: BUCKET/experimental_deform/mini_v1')
    parser.add_argument('--weights', required=True, help='Exported numeric_weights directory with hashed provenance')
    parser.add_argument('--episodes', type=int, default=5, help='First N complete episodes; all their frames are processed')
    parser.add_argument('--reference-policy', required=True, choices=['episode-first-candidate'])
    parser.add_argument('--channel-order', required=True, choices=['RGB', 'BGR'], help='Explicit unverified input convention')
    parser.add_argument('--xy-mode', required=True, choices=['normalized_row_col', 'legacy_zero'])
    parser.add_argument('--preview-scale', type=float, default=20)
    parser.add_argument('--identity-policy', choices=['strict', 'declared_constant_area'], default='strict',
                        help='Opt-in static area identity from proven all-fill Zarr metadata; source validity is preserved')
    parser.add_argument('--source-archive-index', help='Complete original archive index, never the episode-selected sparse index; required for declared_constant_area')
    args = parser.parse_args()
    if args.episodes < 1 or not np.isfinite(args.preview_scale) or args.preview_scale <= 0:
        parser.error('Positive episodes and finite fixed preview scale are required')
    cv2.setNumThreads(1)
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
