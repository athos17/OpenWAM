"""Probe the actual loss dispatcher with trainable coupled stand-ins."""
from types import SimpleNamespace

import pytest
import torch

from openwam.model.architectures.base import BaseWAMArchitecture


class Probe(torch.nn.Module):
    compute_loss=BaseWAMArchitecture.compute_loss
    supports_video_only_training=True
    def __init__(self):
        super().__init__()
        self.video=torch.nn.Parameter(torch.tensor(2.))
        self.action=torch.nn.Parameter(torch.tensor(3.))
        self.proprio=torch.nn.Parameter(torch.tensor(5.))
        scheduler=SimpleNamespace(timesteps=torch.tensor([0.]),sigmas=torch.tensor([0.]),
            add_noise=lambda a,n,s:a,training_target=lambda a,n:n-a)
        self.video_backbone=SimpleNamespace(scheduler=scheduler)
        self.action_backbone=SimpleNamespace(scheduler=scheduler)
        self.dtype=torch.float32;self.device=torch.device('cpu');self.calls=[]
    def forward(self,actions,action_timestep,proprio=None,**kw):
        strict=kw.get('_strict_video_only',False)
        self.calls.append((len(kw['latents']),strict))
        assert not strict or (actions is None and proprio is None)
        value=self.video+(self.action+self.proprio if actions is not None else 0)
        video=torch.ones_like(kw['latents'])*value
        return video,None if actions is None else torch.ones_like(actions)*self.action
    def _compute_video_loss(self,pred,*args):return pred.mean()
    def _compute_action_loss(self,pred,*args):return pred.mean()


def inputs(flags):
    flags=torch.tensor(flags,dtype=torch.bool);b=len(flags)
    return dict(input_latents=torch.zeros((b,1,3,1,1)),actions=torch.zeros((b,32,80)),
                proprio=torch.zeros((b,80)),action_is_pad=flags[:,None,None].expand(b,32,80),
                proprio_mask=(~flags)[:,None,None].expand(b,1,80),video_only_mask=flags)


def test_ego_has_no_action_or_proprio_gradients_even_when_trainable():
    m=Probe();r=m.compute_loss(**inputs([True,True]));r['loss'].backward()
    assert r['loss'].item()==2 and r['loss_action'].item()==0
    assert m.video.grad.item()==1 and m.action.grad is None and m.proprio.grad is None
    assert m.calls==[(2,True)]


def test_mixed_dispatch_preserves_sample_mean_and_robot_path():
    m=Probe();r=m.compute_loss(**inputs([True,False,True]));r['loss'].backward()
    # Each ego contributes V=2; robot contributes V+A+P+A=13.
    assert r['loss'].item()==pytest.approx(17/3)
    assert m.video.grad.item()==pytest.approx(1)
    assert m.action.grad.item()==pytest.approx(2/3)
    assert m.proprio.grad.item()==pytest.approx(1/3)
    assert m.calls==[(1,False),(2,True)]


def test_robot_all_invalid_action_mask_is_not_implicitly_video_only():
    m=Probe();data=inputs([False]);data['action_is_pad']=torch.ones((1,32,80),dtype=torch.bool)
    r=m.compute_loss(**data);r['loss'].backward()
    assert m.calls==[(1,False)] and m.action.grad is not None


def test_inconsistent_video_only_supervision_rejected():
    data=inputs([True]);data['proprio_mask']=torch.ones((1,1,80),dtype=torch.bool)
    with pytest.raises(ValueError,match='inconsistent proprio_mask'):
        Probe().compute_loss(**data)
