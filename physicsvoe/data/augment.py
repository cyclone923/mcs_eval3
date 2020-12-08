import json
import random
from argparse import ArgumentParser
from pathlib import Path

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--scenes', type=Path, default=Path('./data/thor/scenes'))
    parser.add_argument('--anom', action='store_true')
    return parser

def main(scenes, anom):
    all_scenes = list(scenes.glob('*.json'))
    all_scenes = [s for s in all_scenes if 'aug' not in s.name]
    random.shuffle(all_scenes)
    for scene_path in all_scenes:
        print(scene_path)
        scene = read_scene(scene_path)
        new_scene = scene.copy()
        if anom:
            anom_fn, anom_name = choose_anom()
            out_path = anom_path(scene_path, anom_name)
            new_scene = anom_fn(new_scene)
        else:
            out_path = augmented_path(scene_path, 1)
            new_scene = augment_scene(new_scene)
        if new_scene:
            write_scene(out_path, new_scene)

def augment_scene(scene):
    new_scene = remove_structures(scene)
    new_scene = randomize_pos(scene)
    new_scene = randomize_init_force(scene)
    return new_scene

def choose_anom():
    fn_info = [(add_hide, 'hide'), (add_force, 'force'), (add_resize, 'resize')]
    return random.choice(fn_info)

def remove_structures(scene):
    is_struct = lambda o: o.get('structure', False)
    scene['objects'] = [o for o in scene['objects'] if not is_struct(o)]
    return scene

def jitter_xyz(p, amt):
    v = lambda: 2*(random.random()-0.5)*amt
    p['x'] += v()
    p['y'] += v()
    p['z'] += v()
    return p

def randomize_pos(scene, amt=2):
    def jitter_start(o):
        init_pos = o['shows'][0]['position']
        new_pos = jitter_xyz(init_pos, amt)
        new_pos['y'] = max(0, new_pos['y'])
        o['shows'][0]['position'] = new_pos
        return o
    scene['objects'] = [jitter_start(o) for o in scene['objects']]
    return scene

def randomize_init_force(scene, amt=100):
    start_time = lambda o: o['shows'][0]['stepBegin']
    def jitter_force(o):
        start = start_time(o)
        if 'forces' in o:
            for f in o['forces']:
                if f['stepBegin'] == start:
                    new_vec = jitter_xyz(f['vector'], amt)
                    f['vector'] = new_vec
        else:
            zero_vec = {k:0.0 for k in ('x', 'y', 'z')}
            new_vec = jitter_xyz(zero_vec, amt)
            force = {'stepBegin': start, 'stepEnd': start, 'vector': new_vec}
            o['forces'] = [force]
        return o
    scene['objects'] = [jitter_force(o) for o in scene['objects']]
    return scene

def add_hide(scene):
    num_objs = len(scene['objects'])
    valid_idxs = list(range(num_objs))
    valid_idxs = [i for i in range(num_objs) if scene['objects'][i].get('structure', False) == False]
    if len(valid_idxs) == 0:
        return None
    obj_idx = random.choice(valid_idxs)
    anom_obj = scene['objects'][obj_idx]
    start_time = anom_obj['shows'][0]['stepBegin']
    hide_time = start_time + random.randint(10, 40)
    scene['objects'][obj_idx]['hides'] = [{'stepBegin': hide_time}]
    return scene

def rand_force(mag):
    _coord = lambda: random.choice([-1, 1])*(0.3+random.random()*0.7)
    return [mag*_coord() for _ in range(3)]

def add_force(scene):
    num_objs = len(scene['objects'])
    valid_idxs = list(range(num_objs))
    valid_idxs = [i for i in range(num_objs) if scene['objects'][i].get('structure', False) == False]
    if len(valid_idxs) == 0:
        return None
    obj_idx = random.choice(valid_idxs)
    anom_obj = scene['objects'][obj_idx]
    start_time = anom_obj['shows'][0]['stepBegin']
    anom_force = dict(zip('xyz', rand_force(300)))
    force_time = start_time + random.randint(10, 40)
    if 'forces' not in anom_obj:
        scene['objects'][obj_idx]['forces'] = []
    scene['objects'][obj_idx]['forces'] += [{'stepBegin': force_time, 'stepEnd': force_time, 'vector': anom_force}]
    return scene

def add_resize(scene):
    num_objs = len(scene['objects'])
    valid_idxs = list(range(num_objs))
    valid_idxs = [i for i in range(num_objs) if scene['objects'][i].get('structure', False) == False]
    if len(valid_idxs) == 0:
        return None
    obj_idx = random.choice(valid_idxs)
    anom_obj = scene['objects'][obj_idx]
    start_time = anom_obj['shows'][0]['stepBegin']
    resize_time = start_time + random.randint(10, 40)
    resize_amt = random.choice([0.3, 2.5])
    sz = dict(zip('xyz', [resize_amt]*3))
    scene['objects'][obj_idx]['resizes'] = [{'stepBegin': resize_time, 'stepEnd': resize_time, 'size': sz}]
    return scene

def augmented_path(path, id):
    parent = path.parent
    new_name = path.stem+f'_AUG_{id:02d}'
    return parent/(new_name+path.suffix)

def anom_path(path, type_):
    parent = path.parent
    new_name = path.stem+f'_ANOM_{type_}'
    return parent/(new_name+path.suffix)

def read_scene(path):
    with path.open('r') as fd:
        data = json.load(fd)
    return data

def write_scene(path, scene):
    with path.open('w') as fd:
        json.dump(scene, fd)

if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.scenes, args.anom)
