"""Prepare spatial cube/label pairs without inventing a physical spectral axis."""
from pathlib import Path

import numpy as np

from .io import digest, inside, pinned, read_json, write_json
from .prepare import _read


def canonical_labels(cube, source_labels, mapping, class_names, invalid_values,
                     excluded_measurements=(), empty_pixel_all_zero=True):
    if cube.ndim != 3 or cube.dtype.kind != 'f':
        raise ValueError('Expected a floating HWC cube')
    if (source_labels.shape != cube.shape[:2] or source_labels.dtype.kind not in 'iuf'
            or not np.isfinite(source_labels).all()
            or not np.equal(source_labels, np.floor(source_labels)).all()):
        raise ValueError('Expected an integer HW label map matching the cube')
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError('Class names must be nonempty and unique')
    parsed = {int(k): v for k, v in mapping.items()}
    if (len(parsed) != len(mapping) or parsed.get(0) != -1
            or any(type(v) is not int or v < -1 or v >= len(class_names)
                   for v in parsed.values())
            or any(k != 0 and v == -1 for k, v in parsed.items())):
        raise ValueError('Invalid mapping; only source zero denotes unlabeled')
    if not set(np.unique(source_labels)).issubset(parsed):
        raise ValueError('Missing source class in mapping')
    if any(type(v) not in (int, float) or not np.isfinite(v) for v in invalid_values):
        raise ValueError('Invalid sentinel list')
    valid = np.isfinite(cube)
    for value in invalid_values:
        valid &= cube != value
    if empty_pixel_all_zero:
        valid[np.all(cube == 0, axis=-1)] = False
    for exclusion in excluded_measurements:
        index = exclusion['index_hwc']
        if (len(index) != 3 or any(type(i) is not int or not 0 <= i < size
                                  for i, size in zip(index, cube.shape))):
            raise ValueError('Excluded measurement is outside the cube')
        if float(cube[tuple(index)]) != exclusion['expected_value']:
            raise ValueError('Excluded measurement no longer matches its source')
        valid[tuple(index)] = False
    labels = np.full(source_labels.shape, -1, dtype=np.int32)
    for source, target in parsed.items():
        labels[source_labels == source] = target
    labels[~valid.any(axis=-1)] = -1
    return labels, valid


def verify_pairs(path):
    """Verify portable artifacts after copying them; no Torch or training imports."""
    path = Path(path)
    doc = read_json(path)
    if doc.get('format') == 'spatial_cube_label_pairs':
        reports, seen = [], set()
        for item in doc['pairs']:
            report = verify_pairs(pinned(path.parent, {'path': item['path'], 'sha256': item['sha256']}))[0]
            if (report['scene_id'] in seen or report['scene_id'] != item['scene_id']
                    or report['class_names'] != doc['class_names'] or report['role'] != item['role']):
                raise ValueError('Inconsistent collection entry')
            seen.add(report['scene_id'])
            reports.append(report)
        return reports
    if doc.get('format') != 'spatial_cube_label_pair' or doc.get('pair_ready') is not True:
        raise ValueError('Expected a complete spatial pair or collection')
    arrays = {}
    try:
        for key in ('cube', 'labels', 'source_labels', 'valid'):
            arrays[key] = np.load(pinned(path.parent, doc[key]), mmap_mode='r', allow_pickle=False)
        cube, labels, source, valid = (arrays[k] for k in ('cube', 'labels', 'source_labels', 'valid'))
        if (list(cube.shape) != doc['shape_hwc'] or cube.ndim != 3 or cube.dtype.str != doc['dtype']
                or labels.shape != cube.shape[:2] or source.shape != labels.shape
                or labels.dtype.kind not in 'iu' or source.dtype.kind not in 'iuf'
                or valid.shape != cube.shape or valid.dtype != bool or doc['ignore_index'] != -1):
            raise ValueError('Pair shapes, types or ignore index disagree')
        if not np.isfinite(source).all() or not np.equal(source, np.floor(source)).all():
            raise ValueError('Source label map is not finite integer-valued data')
        names, mapping = doc['class_names'], {int(k): v for k, v in doc['source_to_class'].items()}
        if (not names or len(names) != len(set(names)) or mapping.get(0) != -1
                or not set(np.unique(source)).issubset(mapping)
                or any(v < -1 or v >= len(names) for v in mapping.values())):
            raise ValueError('Invalid taxonomy')
        expected = np.full(labels.shape, -1, dtype=np.int32)
        for original, canonical in mapping.items():
            expected[source == original] = canonical
        invalid_count = 0
        for start in range(0, cube.shape[0], 16):
            part, mask = cube[start:start+16], valid[start:start+16]
            if np.any(mask & ~np.isfinite(part)):
                raise ValueError('Nonfinite measurement marked valid')
            expected[start:start+16][~mask.any(axis=-1)] = -1
            invalid_count += int(mask.size - mask.sum())
        if not np.array_equal(labels, expected):
            raise ValueError('Label map differs from the declared mapping or validity')
        ids, counts = np.unique(labels, return_counts=True)
        actual = dict(zip(map(str, ids), map(int, counts)))
        if actual != doc['label_counts'] or invalid_count != doc['invalid_measurements']:
            raise ValueError('Pair statistics disagree with its arrays')
        return [{'scene_id': doc['scene_id'], 'role': doc['role'], 'shape_hwc': list(cube.shape),
                 'class_names': names, 'label_counts': actual, 'invalid_measurements': invalid_count,
                 'verified': True}]
    finally:
        for array in arrays.values():
            array._mmap.close()


def rasterize_labels(xy, labels, shape_hw, coordinate_base, missing_class_ids=(39,)):
    """Plebani x,y -> HW. Missing coordinates are never placed at a real pixel."""
    if coordinate_base not in (0, 1):
        raise ValueError('An explicitly verified coordinate base is required')
    xy, labels = np.asarray(xy), np.asarray(labels).reshape(-1)
    if (xy.shape != (len(labels), 2) or xy.dtype.kind not in 'iu'
            or labels.dtype.kind not in 'iu' or np.any(labels <= 0)):
        raise ValueError('Expected integer XY coordinates and positive raw class IDs')
    missing = np.all(xy == 0, axis=1)
    if not np.isin(labels[missing], missing_class_ids).all():
        raise ValueError('Undeclared missing spatial coordinates')
    # Subtract in a signed type so zero in unsigned MATLAB arrays cannot wrap.
    coords, ids = xy[~missing].astype(np.int64) - coordinate_base, labels[~missing]
    h, w = shape_hw
    if np.any(coords < 0) or np.any(coords[:, 0] >= w) or np.any(coords[:, 1] >= h):
        raise ValueError('Coordinate lies outside the cube')
    gt = np.zeros((h, w), dtype=np.int32)
    for (x, y), label in zip(coords, ids):
        if gt[y, x] not in (0, label):
            raise ValueError('Conflicting labels at one pixel')
        gt[y, x] = label
    return gt


def prepare_plebani_pair(recipe_path, output_relative, root):
    """A pinned, registered ENVI BIL cube plus sparse Plebani label map."""
    from scipy.io import loadmat

    root = Path(root)
    recipe = read_json(recipe_path)
    if (recipe.get('schema_version') != 1 or recipe.get('layout') != 'bil_float32_le'
            or recipe.get('coordinate_registration', {}).get('verified') is not True):
        raise ValueError('A verified registration and explicit cube layout are required')
    image = pinned(root, recipe['cube'])
    pinned(root, recipe['header'])
    mat = pinned(root, recipe['labels_mat'])
    h, w, c = recipe['shape_hwc']
    if image.stat().st_size != h * w * c * 4:
        raise ValueError('Cube length differs from the declared BIL layout')
    raw = np.memmap(image, dtype='<f4', mode='r', shape=(h, c, w))
    cube = raw.transpose(0, 2, 1)
    try:
        d = loadmat(mat, variable_names=['im_names', 'pixims', 'pixcrds', 'pixlabs'])
        suffix = int(recipe['observation_suffix'], 16)
        indices = [i for i, name in enumerate(d['im_names'].ravel(), 1)
                   if int(str(name).strip(), 16) == suffix]
        if len(indices) != 1:
            raise ValueError('Observation must occur exactly once in im_names')
        selected = d['pixims'].ravel() == indices[0]
        gt = rasterize_labels(d['pixcrds'][selected], d['pixlabs'].ravel()[selected],
                              (h, w), recipe['coordinate_registration']['base'])
        if np.count_nonzero(gt) != recipe['expected_labeled_pixels']:
            raise ValueError('Registered label count differs from the recipe')
        taxonomy_path = inside(root, recipe['taxonomy'])
        taxonomy = read_json(taxonomy_path)
        mapping = {'0': -1, **taxonomy['plebani_raw_to_dense']}
        labels, valid = canonical_labels(cube, gt, mapping, taxonomy['class_names'], [65535])
        good = np.asarray(recipe['good_channels'], dtype=bool)
        if good.shape != (c,):
            raise ValueError('Band quality mask has the wrong length')
        valid &= good[None, None, :]
        labels[~valid.any(axis=-1)] = -1
        output = inside(root, output_relative)
        output.mkdir(parents=True, exist_ok=False)
        refs = {}
        for name, array in (('cube', cube), ('labels', labels), ('source_labels', gt),
                            ('valid', valid)):
            path = output / (name + '.npy')
            np.save(path, array, allow_pickle=False)
            refs[name] = {'path': path.name, 'sha256': digest(path)}
        ids, counts = np.unique(labels, return_counts=True)
        document = {'schema_version': 1, 'format': 'spatial_cube_label_pair',
                    'scene_id': recipe['scene_id'], 'role': 'development_candidate',
                    'pair_ready': True, 'shape_hwc': [h, w, c], 'dtype': cube.dtype.str,
                    'class_names': taxonomy['class_names'], 'ignore_index': -1, **refs,
                    'source_to_class': mapping,
                    'label_counts': dict(zip(map(str, ids), map(int, counts))),
                    'invalid_measurements': int(valid.size - valid.sum()),
                    'spectral_axis': recipe['spectral_axis'],
                    'spatial_context': 'real',
                    'provenance': {'recipe': recipe, 'recipe_sha256': digest(recipe_path),
                                   'taxonomy_sha256': digest(taxonomy_path),
                                   'values_changed': False, 'geometry_changed': False}}
        write_json(output / 'pair.json', document)
        return output
    finally:
        raw._mmap.close()


def prepare_pairs(recipe_path, output_relative, root):
    """New directory only. Original values and geometry are preserved verbatim."""
    root = Path(root)
    recipe = read_json(recipe_path)
    if recipe.get('schema_version') != 1:
        raise ValueError('Unsupported pair recipe')
    taxonomy_path = inside(root, recipe['taxonomy'])
    taxonomy = read_json(taxonomy_path)
    names = taxonomy['class_names']
    scenes = recipe['scenes']
    scene_ids = [s['scene_id'] for s in scenes]
    development = recipe['development_scene_ids']
    held_out = recipe['held_out_scene_id']
    if (len(scene_ids) != len(set(scene_ids)) or len(development) != 2
            or len(set(development)) != 2 or held_out in development
            or set(scene_ids) != set(development) | {held_out}):
        raise ValueError('Expected two development scenes and one separate held-out scene')
    output = inside(root, output_relative)
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'format': 'spatial_cube_label_pairs',
              'class_names': names, 'ignore_index': -1,
              'development_scene_ids': development, 'held_out_scene_id': held_out,
              'recipe_sha256': digest(recipe_path),
              'taxonomy_sha256': digest(taxonomy_path), 'pairs': []}
    for scene in scenes:
        scene_id = scene['scene_id']
        folder = inside(output, scene_id)
        cube = _read(root, scene['cube'])
        gt = _read(root, scene['labels'])
        if list(cube.shape) != scene['expected_shape_hwc']:
            raise ValueError(f'Unexpected cube shape: {scene_id}')
        mapping = taxonomy['hymars_local_to_dense'][scene_id]
        labels, valid = canonical_labels(
            cube, gt, mapping, names, scene['invalid_values'],
            scene.get('excluded_measurements', []), scene['empty_pixel_all_zero'])
        folder.mkdir()
        refs = {}
        for name, array in (('cube', cube), ('labels', labels),
                            ('source_labels', gt), ('valid', valid)):
            path = folder / (name + '.npy')
            np.save(path, array, allow_pickle=False)
            refs[name] = {'path': path.name, 'sha256': digest(path)}
        ids, counts = np.unique(labels, return_counts=True)
        pair = {'schema_version': 1, 'format': 'spatial_cube_label_pair',
                'scene_id': scene_id,
                'role': 'held_out_test' if scene_id == held_out else 'development',
                'pair_ready': True, 'shape_hwc': list(cube.shape),
                'dtype': cube.dtype.str, 'class_names': names, 'ignore_index': -1,
                **refs, 'source_to_class': mapping,
                'label_counts': dict(zip(map(str, ids), map(int, counts))),
                'invalid_measurements': int(valid.size - valid.sum()),
                'spectral_axis': None, 'spectral_axis_status': 'pending_channel_mapping',
                'spatial_context': 'real',
                'provenance': {'sources': scene, 'values_changed': False,
                               'geometry_changed': False,
                               'taxonomy_sha256': result['taxonomy_sha256']}}
        write_json(folder / 'pair.json', pair)
        result['pairs'].append({'scene_id': scene_id, 'role': pair['role'],
                                'path': f'{scene_id}/pair.json',
                                'sha256': digest(folder / 'pair.json')})
        del cube, gt, labels, valid, array
    # The collection file is the completion marker; failed partial builds lack it.
    write_json(output / 'taxonomy.json', taxonomy)
    write_json(output / 'collection.json', result)
    return output
