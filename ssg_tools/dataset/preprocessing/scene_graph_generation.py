from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
from collections import defaultdict
import logging
import json

log = logging.getLogger(__name__)


def scene_graph_remapping(dataset: DatasetInterface3DSSG,
                          overwrite: bool = False):
    output_path = dataset.path_scenegraph_data
    if output_path.exists():
        if overwrite:
            output_path.unlink()
            dataset = DatasetInterface3DSSG(dataset.path)
        else:
            log.info("Scene graph data exists. Skipping ...")
            return

    log.info("Building scene graph from objects and relationships...")
    path_relationships = dataset.path_3dssg / "relationships.json"
    path_objects = dataset.path_3dssg / "objects.json"
    with open(path_relationships, 'r') as f:
        relationships = json.load(f)
    relationships = {v['scan']: v['relationships'] for v in relationships["scans"] if v['scan'] in dataset.scan_ids}

    with open(path_objects, 'r') as f:
        objects = json.load(f)
        # reorder the data from a list of lists into a dict of dicts for random access
    objects = {scan['scan']: {int(obj["id"]): obj for obj in scan["objects"]} for scan in objects["scans"] if scan['scan'] in dataset.scan_ids}
    
    scene_graph = {k: {'nodes': objects[k], "edges": relationships[k]} for k in relationships.keys()}

    RIO27_CLASSES = ['-', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'counter', 'shelf', 'curtain', 'pillow', 'clothes', 'ceiling', 'fridge', 'tv', 'towel', 'plant', 'box', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'object', 'blanket']

    log.info("Removing unused information from scene graph data...")
    _keys_to_remove_ = set(['nyu40', 'eigen13', 'attributes', 'affordances', 'symmetry', 'state_affordances'])

    for scene_id in scene_graph.keys():
        scene_objs = scene_graph[scene_id]['nodes']
    
        # remove unnecessary annotations & rename object keys to be more mearningful
        for obj_id, obj in scene_objs.items():
            # remove everything that is not needed
            for k in set(obj.keys()).intersection(_keys_to_remove_): del obj[k]
            obj['raw528_enc'] = int(obj.pop('global_id'))
            obj['raw528_name'] = obj.pop('label')
            obj['rio27_enc'] = int(obj.pop('rio27'))
            obj['rio27_name'] = RIO27_CLASSES[obj['rio27_enc']]
            obj['instance_color'] = obj.pop('ply_color')
            obj['instance_id'] = int(obj.pop('id'))
        
        # delete objects assigned with "0 : -" RIO27 annotation
        obj_ids_to_remove = [obj_id for obj_id in scene_objs.keys() if scene_objs[obj_id]['rio27_enc'] == 0]
        for k in obj_ids_to_remove: del scene_objs[k]
            

    log.info("Checking scene graph validity...")
    # unit testing (checking)
    # 1. ensure that all nodes would contain the same node_keys
    # 2. ensure that there is no more objects with "0 : -" RIO27 annotation (27 unique keys)
    for scene_id, scene in scene_graph.items():
        scene_objs = scene['nodes']
        for obj in scene_objs.values():
            assert obj.keys() == {'raw528_name', 'raw528_enc', 'rio27_name', 'instance_id', 'rio27_enc', 'instance_color'}, obj.keys()
            assert 0 < obj['rio27_enc'] < 28, '{}:{}'.format(obj['rio27_enc'], obj['rio27_name'])
            assert not (obj['rio27_name'] == '-'), '{}:{}'.format(obj['rio27_enc'], obj['rio27_name'])


    log.info("Relabelling relationships...")
    #STRUCTURAL_RELATIONSHIP_CLASSES = config.dataset.scene_graph.relationships
    STRUCTURAL_RELATIONSHIP_CLASSES = ['supported by', 'attached to', 'standing on', 'lying on','hanging on', 
                                    'connected to', 'leaning against', 'part of', 'belonging to', 'build in',
                                    'standing in', 'cover', 'lying in', 'hanging in', 'spatial proximity', 'close by']
    for scene_id, scene in scene_graph.items():
        scene_rels = scene['edges']
        
        # remove comparative relationships 
        rel_ids_to_remove = []
        for rel_id, rel_tuple in enumerate(scene_rels):
            rel_name = rel_tuple[-1]
            _rel_names_to_retain_ = ['part of', 'left', 'cover', 'hanging in', 
                                    'belonging to', 'connected to', 'supported by', 
                                    'hanging on', 'right', 'attached to', 'build in', 
                                    'close by', 'behind', 'lying on', 'standing on', 
                                    'lying in', 'standing in', 'front', 'leaning against']
            if rel_name not in _rel_names_to_retain_: rel_ids_to_remove.append(rel_id)
        rel_ids_to_remove.reverse()
        for rel_id in rel_ids_to_remove: del scene_rels[rel_id]
            
        # rename "left | right | front | behind" into "spatial proximity" & reassign rels with class_id (above)
        for rel_id, rel_tuple in enumerate(scene_rels):
            rel_name = rel_tuple[-1]
            rel_name = rel_name.replace('left', 'spatial proximity')  \
                                .replace('right', 'spatial proximity') \
                                .replace('front', 'spatial proximity') \
                                .replace('behind', 'spatial proximity') 
            scene_rels[rel_id][-1] = rel_name
            scene_rels[rel_id][-2] = STRUCTURAL_RELATIONSHIP_CLASSES.index(rel_name)

        # aggregate multi-label edges -> merge multiple 'spatial proximity' -> reformulate from 'multi-label' to 'multi-class'
        rel_aggregation = defaultdict(list)
        for rel_tuple in scene_rels:
            src_node, dst_node, rel_enc, rel_name = rel_tuple
            rel_aggregation['{}-{}'.format(src_node, dst_node)].append('{}-{}'.format(rel_enc, rel_name))
        for rel_key, rel_value in rel_aggregation.items():
            rel_value = set(rel_value)
            rel_aggregation[rel_key] = rel_value
            if len(rel_value) == 1: continue # normal case - multi-class setting already
            
            if len(rel_value) == 2 and '15-close by' in rel_value:
                # case 1
                rel_value.remove('15-close by')
            elif len(rel_value) == 2 and '14-spatial proximity' in rel_value:
                rel_value.remove('14-spatial proximity')
            elif len(rel_value) == 3 and '15-close by' in rel_value and'14-spatial proximity' in rel_value:
                rel_value.remove('15-close by')
                rel_value.remove('14-spatial proximity')
            else:
                print('Error - there should be no more cases')
                exit()
            rel_aggregation[rel_key] = rel_value
        
        # recover it back to the list of edge_tuple shape
        new_scene_rels = []
        for rel_key, rel_value in rel_aggregation.items():
            assert len(rel_value) == 1, 'unsuccessful reformulation'  # unit testing
            src_node, dst_node = rel_key.split('-')
            rel_enc, rel_name = list(rel_value)[0].split('-')
            curr_tuple = [int(src_node), int(dst_node), int(rel_enc), rel_name]
            new_scene_rels.append(curr_tuple)

        scene['edges'] = new_scene_rels

    log.info("Recalibration nodes and edges in the scene graph...")

    for scene_id, scene in scene_graph.items():
        scene_objs = scene['nodes']
        scene_rels = scene['edges']
    
        # remove edges connecting to invalid-nodes
        valid_scene_obj_ids = [int(i) for i in list(scene_objs.keys())]
        invalid_scene_rel_ids = []
        for rel_id, rel_tuple in enumerate(scene_rels):
            src_node, dst_node = rel_tuple[0], rel_tuple[1]
            if (src_node not in valid_scene_obj_ids) or (dst_node not in valid_scene_obj_ids):
                invalid_scene_rel_ids.append(rel_id)
        invalid_scene_rel_ids.reverse()
        for invalid_rel_id in invalid_scene_rel_ids: del scene_rels[invalid_rel_id]
            
        # remove isolated nodes with no edges connected
        appeared_obj_ids_in_edges = []
        isolated_scene_obj_ids = []
        for rel_id, rel_tuple in enumerate(scene_rels):
            src_node, dst_node = rel_tuple[0], rel_tuple[1]
            appeared_obj_ids_in_edges.append(src_node)
            appeared_obj_ids_in_edges.append(dst_node)
        appeared_obj_ids_in_edges = set(appeared_obj_ids_in_edges)
        for obj_id in scene_objs.keys():
            if obj_id not in appeared_obj_ids_in_edges:
                isolated_scene_obj_ids.append(obj_id)
        for isolated_obj_id in isolated_scene_obj_ids: del scene_objs[isolated_obj_id]

    scene_ids_to_remove = ['a8952593-9035-254b-8f40-bc82e6bcbbb1',
                        '20c993b9-698f-29c5-87f1-4514b70070c3',
                        '20c99397-698f-29c5-8534-5304111c28af',
                        '20c993c7-698f-29c5-8685-0d1a2a4a3496',
                        'ae73fa15-5a60-2398-8646-dd46c46a9a3d',
                        '20c993c5-698f-29c5-8604-3248ede4091f',
                        '6bde60cd-9162-246f-8fad-fca80b4d6ad8',
                        '77941464-cfdf-29cb-87f4-0465d3b9ab00',
                        '0cac75af-8d6f-2d13-8f9e-ed3f62665aed',
                        '0cac768a-8d6f-2d13-8dd3-3cbb7d916641',
                        'ba6fda98-a4c1-2dca-8230-bce60f5a0f85',
                        'd7d40d48-7a5d-2b36-97ad-692c9b56b508',
                        'd7d40d46-7a5d-2b36-9734-659bccb1c202',
                        '352e9c48-69fb-27a7-8a35-3dbf699637c8',
                        'ba6fdaa0-a4c1-2dca-80a9-df196c04fcc8',
                        'd7d40d40-7a5d-2b36-977c-4e35fdd5f03a',
                        '0cac75e6-8d6f-2d13-8e4a-72b0fc5dc6c3',
                        '38770cab-86d7-27b8-85cd-d55727c4697b',
                        '0cac768c-8d6f-2d13-8cc8-7ace156fc3e7']
    
    # one special scene who contains no elements after our preprocessing
    for scene_id, scene in scene_graph.items():
        scene_objs = scene['nodes']
        scene_rels = scene['edges']
        len_objs = len(list(scene_objs.keys()))
        len_rels = len(scene_rels)
        if len_objs * len_rels == 0:
            print(scene_id, '- len_objects: ', len_objs, '& len_relationships: ',len_rels)
            scene_ids_to_remove.append(scene_id)

    scene_ids_to_remove = set(scene_ids_to_remove).intersection(scene_graph.keys())
    log.info("Removing %d partial or invalid scenes from the dataset...", len(scene_ids_to_remove))
    for scene_id_to_remove in scene_ids_to_remove: del scene_graph[scene_id_to_remove]

    # store the number of classes for both edges and nodes
    scene_graph_data = {"scenes": scene_graph, 
                        "edge_classes": STRUCTURAL_RELATIONSHIP_CLASSES,
                        "node_classes": RIO27_CLASSES}

    log.info("Saving processed scene graph...")
    with open(output_path, 'w') as f:
        json.dump(scene_graph_data, f, indent=4)
    log.info("done.")

# def main():
#     from ssg_tools import ConfigArgumentParser, init_logging

#     parser = ConfigArgumentParser(
#         description='Generate the scene graph from the raw 3DSSG dataset.')
#     parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files.')
#     args = parser.parse_args()

#     config = args.config
#     init_logging(config)

#     dataset = DatasetInterface3DSSG(config)
#     scene_graph_remapping(dataset, overwrite=args.overwrite)

# if __name__ == "__main__":
#     main()