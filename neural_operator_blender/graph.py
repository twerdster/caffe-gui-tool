"""Portable, data-only neural graph interchange and validation."""
import heapq
import json
import math
import os
import tempfile
from pathlib import Path
import bpy

SCHEMA = 'neural-operator-composer/v1'
MAX_NODES = 10000
MAX_LINKS = 50000
MAX_SOCKETS = 1024
MAX_TOTAL_SOCKETS = 50000
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_STRING_BYTES = 1024 * 1024


def _object(value, path, allowed=None, required=()):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(path + ' must be an object with string keys')
    if allowed is not None and set(value) - set(allowed):
        raise ValueError(path + ' has unknown fields: ' + ', '.join(sorted(set(value) - set(allowed))))
    missing = set(required) - set(value)
    if missing:
        raise ValueError(path + ' is missing fields: ' + ', '.join(sorted(missing)))


def _string(value, path, nonempty=False, limit=MAX_STRING_BYTES):
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(path + ' must be ' + ('a nonempty string' if nonempty else 'a string'))
    try:
        encoded = value.encode('utf-8')
    except UnicodeError as error:
        raise ValueError(path + ' must contain valid Unicode') from error
    if '\x00' in value or len(encoded) > limit:
        raise ValueError(path + ' contains a null character or exceeds its byte limit')


def _number(value, path, minimum=-3.4028234663852886e38, maximum=3.4028234663852886e38):
    if type(value) not in (int, float) or not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(path + ' must be a finite number within its allowed bounds')


def _vector(value, path, size, minimum=-3.4028234663852886e38, maximum=3.4028234663852886e38):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(path + ' must contain ' + str(size) + ' numbers')
    for index, item in enumerate(value):
        _number(item, path + '/' + str(index), minimum, maximum)


def _check_fields(value, fields, path='properties'):
    _object(fields, path, property_names(value))
    for key, field in fields.items():
        prop = value.bl_rna.properties[key]
        location = path + '/' + key
        if prop.type in {'POINTER', 'COLLECTION'}:
            item_class = bpy.types.PropertyGroup.bl_rna_get_subclass_py(prop.fixed_type.identifier)
            if item_class is None:
                raise ValueError('Unsupported property: ' + location)
            if prop.type == 'POINTER':
                _check_fields(item_class, field, location)
            else:
                if not isinstance(field, list) or len(field) > MAX_SOCKETS:
                    raise ValueError(location + ' must be an array within the collection size limit')
                for item in field:
                    _check_fields(item_class, item, location)
        elif prop.type == 'BOOLEAN':
            if type(field) is not bool:
                raise ValueError(location + ' must be a boolean')
        elif prop.type == 'INT':
            if type(field) is not int or not prop.hard_min <= field <= prop.hard_max:
                raise ValueError(location + ' must be an integer within its allowed bounds')
            if key in {'input_amount', 'output_amount', 'num_of_slices'} and field > MAX_SOCKETS:
                raise ValueError(location + ' exceeds the socket count limit')
        elif prop.type == 'FLOAT':
            _number(field, location, prop.hard_min, prop.hard_max)
        elif prop.type == 'ENUM':
            if not isinstance(field, str) or field not in prop.enum_items.keys():
                raise ValueError(location + ' must be one of the registered enum identifiers')
        elif prop.type == 'STRING':
            _string(field, location, limit=prop.length_max - 1 if prop.length_max else MAX_STRING_BYTES)
        else:
            raise ValueError('Unsupported property: ' + location)


def property_names(value):
    result = set()
    for cls in (value if isinstance(value, type) else type(value)).__mro__:
        result.update(getattr(cls, '__annotations__', {}))
    return sorted(result)

def properties(value):
    result = {}
    for key in property_names(value):
        field = getattr(value, key)
        if isinstance(field, bpy.types.PropertyGroup):
            result[key] = properties(field)
        elif key in value.bl_rna.properties and value.bl_rna.properties[key].type == 'COLLECTION':
            result[key] = [properties(item) for item in field]
        elif isinstance(field, (str, int, float, bool)):
            result[key] = field
    return result

def set_properties(value, fields):
    _check_fields(value, fields)
    _set_properties(value, fields)


def _set_properties(value, fields):
    # Socket counts must be applied before their per-socket collections.
    items = sorted(fields.items(), key=lambda item: (item[0] not in {'input_amount', 'output_amount', 'num_of_slices'}, item[0]))
    for key, field in items:
        target = getattr(value, key)
        if isinstance(field, dict):
            _set_properties(target, field)
        elif isinstance(field, list):
            target.clear()
            for item in field:
                _set_properties(target.add(), item)
        else:
            setattr(value, key, field)

def to_dict(tree):
    nodes = []
    for node in tree.nodes:
        nodes.append({
            'id': node.name, 'type': node.bl_idname, 'label': node.label,
            'location': list(node.location), 'width': node.width,
            'hide': node.hide, 'color': list(node.color),
            'use_custom_color': node.use_custom_color,
            'parent': node.parent.name if node.parent else None,
            'properties': properties(node),
            'outputs': [getattr(s, 'output_name', '') for s in node.outputs],
        })
    links = [
        {'from': link.from_node.name, 'output': list(link.from_node.outputs).index(link.from_socket),
         'to': link.to_node.name, 'input': list(link.to_node.inputs).index(link.to_socket)}
        for link in tree.links
    ]
    return {'schema': SCHEMA, 'name': tree.name, 'nodes': nodes, 'links': links}

def from_dict(document, name=None):
    _object(document, 'Graph', {'schema', 'name', 'nodes', 'links'}, {'schema', 'name', 'nodes', 'links'})
    if document['schema'] != SCHEMA:
        raise ValueError('Unsupported neural graph schema')
    _string(document['name'], 'Graph name', nonempty=True, limit=bpy.types.NodeTree.bl_rna.properties['name'].length_max - 1)
    if name is not None:
        _string(name, 'Graph name override', nonempty=True, limit=bpy.types.NodeTree.bl_rna.properties['name'].length_max - 1)
    for key, limit in [('nodes', MAX_NODES), ('links', MAX_LINKS)]:
        if not isinstance(document[key], list) or len(document[key]) > limit:
            raise ValueError('Graph ' + key + ' must be an array within the import size limit')
    from .caffe_nodes import CLASSES
    types = {cls.bl_idname for cls in CLASSES if issubclass(cls, bpy.types.Node)} | {'NodeFrame', 'NodeReroute'}
    records = {}
    for record in document['nodes']:
        _object(record, 'Node', {'id', 'type', 'label', 'location', 'width', 'hide', 'color', 'use_custom_color', 'parent', 'properties', 'outputs'}, {'id', 'type', 'location', 'outputs'})
        _string(record['id'], 'Node identifier', nonempty=True, limit=bpy.types.Node.bl_rna.properties['name'].length_max - 1)
        _string(record['type'], 'Node type', nonempty=True)
        if record['id'] in records:
            raise ValueError('Duplicate node identifiers')
        if record['type'] not in types:
            raise ValueError('Unknown neural node type')
        records[record['id']] = record
        _string(record.get('label', ''), 'Node label', limit=bpy.types.Node.bl_rna.properties['label'].length_max - 1)
        _vector(record['location'], 'Node location', 2)
        _vector(record.get('color', [0.3, 0.3, 0.3]), 'Node color', 3, 0, 1)
        _number(record.get('width', 180), 'Node width', 1)
        for key in ('hide', 'use_custom_color'):
            if type(record.get(key, False)) is not bool:
                raise ValueError('Node ' + key + ' must be a boolean')
        if not isinstance(record['outputs'], list) or len(record['outputs']) > MAX_SOCKETS + 1:
            raise ValueError('Node outputs must be an array within the socket count limit')
        for output in record['outputs']:
            _string(output, 'Output name')
    # Blender silently clears invalid parents/cycles; reject them before changing data.
    for record in document['nodes']:
        parent = record.get('parent')
        if parent is not None and (not isinstance(parent, str) or parent not in records or records[parent]['type'] != 'NodeFrame'):
            raise ValueError('Node parent must refer to a frame in the graph')
    checked = set()
    for identifier in records:
        ancestors = set()
        current = identifier
        while current is not None and current not in checked:
            if current in ancestors:
                raise ValueError('Cycle in frame parents')
            ancestors.add(current)
            current = records[current].get('parent')
        checked.update(ancestors)
    destinations = set()
    for link in document['links']:
        _object(link, 'Link', {'from', 'to', 'output', 'input'}, {'from', 'to', 'output', 'input'})
        if any(not isinstance(link[key], str) or link[key] not in records for key in ('from', 'to')):
            raise ValueError('Link refers to a missing node')
        if any(type(link[key]) is not int or link[key] < 0 for key in ('output', 'input')):
            raise ValueError('Socket indices must be nonnegative integers')
        destination = (link['to'], link['input'])
        if destination in destinations:
            raise ValueError('Multiple links target the same input socket')
        destinations.add(destination)
    tree = bpy.data.node_groups.new(name if name is not None else document['name'], 'CaffeNodeTree')
    tree.use_fake_user = True
    try:
        nodes = {}
        socket_total = 0
        for record in document['nodes']:
            node = tree.nodes.new(record['type'])
            node.name = record['id']
            if node.name != record['id']:
                raise ValueError('Node identifier could not be retained: ' + record['id'])
            node.label = record.get('label', '')
            set_properties(node, record.get('properties', {}))
            socket_total += len(node.inputs) + len(node.outputs)
            if socket_total > MAX_TOTAL_SOCKETS:
                raise ValueError('Graph exceeds the total socket count limit')
            width = record.get('width', 180)
            node.width = width
            if not math.isclose(node.width, width, rel_tol=1e-6, abs_tol=1e-4):
                raise ValueError('Node width is outside the supported range: ' + node.name)
            node.hide = record.get('hide', False)
            node.color = record.get('color', (0.3, 0.3, 0.3))
            node.use_custom_color = record.get('use_custom_color', False)
            output_names = record['outputs']
            if len(output_names) != len(node.outputs):
                raise ValueError('Output names must match the node socket count: ' + node.name)
            for socket, text in zip(node.outputs, output_names):
                if hasattr(socket, 'output_name'):
                    socket.output_name = text
                elif text:
                    raise ValueError('This socket does not support an output name: ' + node.name)
            nodes[record['id']] = node
        for record in document['nodes']:
            node = nodes[record['id']]
            if record.get('parent') is not None:
                node.parent = nodes[record['parent']]
            node.location = record['location']
        for link in document['links']:
            nodes_from, nodes_to = nodes[link['from']], nodes[link['to']]
            if link['output'] >= len(nodes_from.outputs) or link['input'] >= len(nodes_to.inputs):
                raise ValueError('Link socket index is out of range')
        for link in document['links']:
            nodes_from, nodes_to = nodes[link['from']], nodes[link['to']]
            tree.links.new(nodes_from.outputs[link['output']], nodes_to.inputs[link['input']])
        if len(tree.links) != len(document['links']):
            raise ValueError('Blender could not retain every imported link')
        return tree
    except Exception:
        bpy.data.node_groups.remove(tree)
        raise


def save(tree, path):
    # Compute valid JSON before touching the destination. Replace it only after
    # the entire payload has been written, retaining the previous graph on error.
    payload = json.dumps(to_dict(tree), indent=2, sort_keys=True, allow_nan=False) + '\n'
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def load(path):
    with Path(path).open('rb') as stream:
        payload = stream.read(MAX_FILE_BYTES + 1)
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError('Graph JSON exceeds the file size limit')
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    try:
        document = json.loads(payload.decode('utf-8'), parse_constant=invalid_constant, object_pairs_hook=_unique_object)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('Graph JSON is not valid UTF-8 or is nested too deeply') from error
    return from_dict(document)

def topological(tree):
    if getattr(tree, 'bl_idname', None) != 'CaffeNodeTree':
        raise ValueError('Expected a Caffe neural node tree')
    nodes = {n.name: n for n in tree.nodes if n.bl_idname not in {'NodeFrame', 'SolverNodeType'}}
    predecessors = {name: set() for name in nodes}
    successors = {name: set() for name in nodes}
    for link in tree.links:
        source, target = link.from_node.name, link.to_node.name
        if source not in nodes or target not in nodes:
            raise ValueError('Link refers to a non-processing node')
        predecessors[target].add(source)
        successors[source].add(target)
    ready = [name for name in nodes if not predecessors[name]]
    heapq.heapify(ready)
    ordered = []
    while ready:
        name = heapq.heappop(ready)
        ordered.append(nodes[name])
        for target in sorted(successors[name]):
            predecessors[target].remove(name)
            if not predecessors[target]:
                heapq.heappush(ready, target)
    if len(ordered) != len(nodes):
        remaining = sorted(name for name in nodes if predecessors[name])
        raise ValueError('Cycle in neural graph: ' + ', '.join(remaining[:5]))
    return ordered

def validate(tree):
    ordered = topological(tree)
    blobs, errors = {}, []
    if not ordered:
        errors.append('Graph has no neural processing or data nodes')
    for link in tree.links:
        if not link.is_valid or link.is_muted:
            errors.append('Invalid or muted graph link: ' + link.from_node.name + ' -> ' + link.to_node.name)
    for node in tree.nodes:
        try:
            _check_fields(node, properties(node), node.name)
        except ValueError as error:
            errors.append(str(error))
    for node in ordered:
        if node.bl_idname == 'NodeReroute':
            errors.append('Replace reroutes with direct links before Caffe export: ' + node.name)
        for socket in node.outputs:
            name = getattr(socket, 'output_name', '')
            if not name or any(c in name for c in '\n\r"\\'):
                errors.append('Set a valid output blob name: ' + node.name)
            elif name in blobs:
                errors.append('Duplicate output blob: ' + name)
            blobs[name] = node.name
        for socket in node.inputs:
            if not socket.is_linked:
                errors.append('Unconnected input: ' + node.name + '/' + socket.name)
            elif len(socket.links) != 1:
                errors.append('Multiple links target the same input socket: ' + node.name + '/' + socket.name)
        if any(c in node.name for c in '\n\r"\\'):
            errors.append('Invalid Caffe layer name: ' + node.name)
    if errors:
        raise ValueError('\n'.join(errors[:15]))
    return {'nodes': len(ordered), 'links': len(tree.links), 'blobs': len(blobs)}
