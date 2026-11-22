"""Spatial split of complete cube/GT pairs, without copying or modifying cubes."""
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from .io import digest, inside, pinned, read_json, write_json


def spatial_masks(labels, scene_id, block_size=128, patch_size=19, seed=42,
                  validation_block_probability=.2):
    if (labels.ndim != 2 or labels.dtype.kind not in 'iu' or not (labels >= 0).any()
            or type(patch_size) is not int or patch_size < 1 or patch_size % 2 != 1
            or type(block_size) is not int or block_size < patch_size
            or not 0 < validation_block_probability < 1):
        raise ValueError('Integer HW labels and a valid spatial block/patch size are required')
    h, w = labels.shape
    footprint = np.ones((patch_size, patch_size), dtype=bool)
    border_eligible = binary_erosion(np.ones((h, w), dtype=bool), footprint, border_value=0)
    local_seed = seed ^ int.from_bytes(sha256(scene_id.encode()).digest()[:4], 'little')
    grid = np.random.default_rng(local_seed).random(
        ((h + block_size - 1) // block_size, (w + block_size - 1) // block_size)) < validation_block_probability
    validation = np.repeat(np.repeat(grid, block_size, 0), block_size, 1)[:h, :w]
    protected = []
    # A tiny class must not disappear entirely into a boundary gap or validation.
    # Protect its real patch footprints; no synthetic samples or relabeling.
    for label in np.unique(labels[labels >= 0]):
        eligible = (labels == label) & border_eligible
        train_centers = binary_erosion(~validation, footprint, border_value=0)
        if eligible.any() and not (train_centers & eligible).any():
            validation[binary_dilation(eligible, footprint)] = False
            protected.append(int(label))
    train = ~validation
    train_centers = binary_erosion(train, footprint, border_value=0) & (labels >= 0)
    validation_centers = binary_erosion(validation, footprint, border_value=0) & (labels >= 0)
    if not train_centers.any() or not validation_centers.any():
        raise ValueError('This block split lacks labeled train or validation centers')
    # Every full footprint lies within one disjoint ROI; shared neighborhoods are impossible.
    if np.any(binary_dilation(train_centers, footprint) &
              binary_dilation(validation_centers, footprint)):
        raise ValueError('Train and validation patch footprints overlap')
    return {'train_roi': train, 'validation_roi': validation,
            'train_centers': train_centers, 'validation_centers': validation_centers}, protected


def prepare_pair_split(recipe_path, output_relative, root):
    root = Path(root)
    recipe = read_json(recipe_path)
    if recipe.get('schema_version') != 1:
        raise ValueError('Unsupported split recipe')
    patch_size = recipe['patch_size']
    if (type(patch_size) is not int or patch_size < 1 or patch_size % 2 != 1
            or type(recipe['block_size']) is not int or recipe['block_size'] < patch_size
            or type(recipe['seed']) is not int or recipe['seed'] < 0):
        raise ValueError('Invalid patch, block size or random seed')
    loaded, scene_ids, class_names = [], set(), None
    for entry in recipe['pairs']:
        path = pinned(root, entry['pair'])
        pair = read_json(path)
        scene_id = pair['scene_id']
        if scene_id in scene_ids or pair.get('pair_ready') is not True:
            raise ValueError('Repeated or unfinished pair')
        scene_ids.add(scene_id)
        class_names = pair['class_names'] if class_names is None else class_names
        if pair['class_names'] != class_names:
            raise ValueError('Class order differs between pairs')
        is_held_out = scene_id == recipe['held_out_scene_id']
        if (entry['stage'] not in ('plebani350', 'hymars_full')
                or entry['role'] not in ('development', 'held_out_test')
                or (entry['role'] == 'held_out_test') != is_held_out
                or (pair['role'] == 'held_out_test') != is_held_out):
            raise ValueError('Held-out scene cannot participate in development')
        loaded.append((entry, pair, path))
    if recipe['held_out_scene_id'] not in scene_ids:
        raise ValueError('The split must contain the declared held-out scene')
    output = inside(root, output_relative)
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'recipe': recipe, 'recipe_sha256': digest(recipe_path),
              'class_names': class_names, 'pairs': [], 'shared_patch_pixels': 0}
    for entry, pair, path in loaded:
        labels = np.load(pinned(path.parent, pair['labels']), mmap_mode='r', allow_pickle=False)
        try:
            if entry['role'] == 'held_out_test':
                roi = np.ones(labels.shape, dtype=bool)
                arrays = {'test_roi': roi, 'test_centers': binary_erosion(
                    roi, np.ones((patch_size, patch_size), dtype=bool), border_value=0) & (labels >= 0)}
                protected = []
            else:
                arrays, protected = spatial_masks(labels, pair['scene_id'],
                    recipe['block_size'], patch_size, recipe['seed'],
                    recipe.get('validation_block_probability', .2))
            folder = inside(output, pair['scene_id'])
            folder.mkdir()
            refs, counts = {}, {}
            for name, array in arrays.items():
                target = folder / (name + '.npy')
                np.save(target, array, allow_pickle=False)
                refs[name] = {'path': f'{pair["scene_id"]}/{target.name}', 'sha256': digest(target)}
                if name.endswith('_centers'):
                    ids, n = np.unique(labels[array], return_counts=True)
                    counts[name] = dict(zip(map(str, ids), map(int, n)))
            result['pairs'].append({'scene_id': pair['scene_id'], 'role': entry['role'],
                'stage': entry['stage'], 'masks': refs, 'labeled_center_counts': counts,
                'classes_protected_in_train': protected,
                'center_counts_scope': 'geometry_and_labels; spectral eligibility checked by training loader'})
        finally:
            labels._mmap.close()
    write_json(output / 'split.json', result)
    return output
