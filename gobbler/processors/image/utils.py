import os.path as path

import gobbler.globals as glb

sys_msg_dsc_diagram = open(
    path.join(glb.instructions_dir, "describe_diagram.txt"), "r"
).read()
sys_msg_dsc_entities = open(
    path.join(glb.instructions_dir, "describe_entity_s.txt"), "r"
).read()
sys_msg_desc_text = open(
    path.join(glb.instructions_dir, "describe_diagram.txt"), "r"
).read()
