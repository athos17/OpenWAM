"""The deployment boundary must preserve optical representation identity."""

from omegaconf import OmegaConf

from openwam.deploy.engine import JointInferenceEngine
from openwam.model.video_backbone.wan.shared.diffusion import FlowMatchScheduler
from openwam.tactile.regions import RGB_CODEC
from tests.test_touchwam_rgb import model as rgb_model, rgb_inputs


def test_deploy_observation_routes_rgb_to_its_actual_generation_head(rgb_model):
    model = rgb_model
    model.video_backbone._scheduler = FlowMatchScheduler("Wan")
    model.normalize_deploy_proprio = lambda x: x
    config = OmegaConf.create({
        "inference": {"denoise_mode": "sync", "denoise_steps": 2, "num_frames": 5, "video_num_frames": 3},
        "optimization": {"decode_video": False, "dit_cache": {"enabled": False}, "compile": {"enabled": False}},
    })
    inputs = rgb_inputs()
    observation = {k: inputs[k][:1] for k in (
        "tactile_maps", "tactile_valid", "tactile_region_ids", "tactile_layout_present", "tactile_spatial_valid")}
    observation.update(tactile_codec=RGB_CODEC, state=inputs["proprio"][:1])
    result = JointInferenceEngine(config, architecture=model).generate({"prompt": "hold", "observation": observation})
    assert result["future_tactile_maps"].shape == (2, 2, 3, 16, 16)
    assert result["tactile_codec"] == RGB_CODEC
    assert result["tactile_representation"] == "optical_rgb"
