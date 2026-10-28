"""Original SharpaDex TCP in its native arm-mount frame, encoded as TCP62.

The global reader provides only transport, masks and normalization; its geometry
was materialized offline. Override its Tianji contract guard for native TCP data.
"""
from openwam.dataloader.sharpadex_tianji_global import (
    MultiSharpaDexTianjiGlobalDataset,
    SharpaDexTianjiGlobalDataset,
)


class MultiSharpaDexNativeTCPDataset(MultiSharpaDexTianjiGlobalDataset):
    @property
    def training_data_contract(self):
        return self.buckets[0].training_data_contract


class SharpaDexNativeTCPDataset(SharpaDexTianjiGlobalDataset):
    DATASET_NAME = 'SharpaDexNativeTCP'

    def _resolve_cameras(self, info):
        expected = {
            'raw_action_dim': 62,
            'storage_vector_dim': 62,
            'unified_action_dim': 80,
            'action_representation': 'absolute_tcp_and_absolute_hand_joints',
            'action_pose_frame': 'sharpadex_arm_mount',
            'state_pose_frame': 'sharpadex_arm_mount',
            'end_effector_frame': 'source_tcp',
            'rotation_6d_layout': '[R00,R10,R20,R01,R11,R21]',
            'geometry_version': 'sharpadex_native_tcp_v1_20261026',
            'target_robot_alignment': False,
            'action_time_alignment': 'recorded_command_at_same_row',
        }
        contract = info.get('openwam_contract', {})
        if info.get('codebase_version') != 'v3.0' or any(contract.get(k) != v for k, v in expected.items()):
            raise ValueError('Expected native SharpaDex TCP62 contract; Tianji-aligned data is incompatible')
        self.training_data_contract = expected.copy()
        return ('observation.images.left_head', 'observation.images.left_wrist', 'observation.images.right_wrist')

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiSharpaDexNativeTCPDataset
