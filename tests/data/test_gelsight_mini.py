"""Numerical checks; no public checkpoint or sensor calibration is implied."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def backend():
    pytest.importorskip('scipy')
    path=Path(__file__).resolve().parents[2]/'scripts/data/ftp1/depth_plugins/gelsight_mini.py'
    spec=importlib.util.spec_from_file_location('gelsight_mini_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def setup_backend():
    m=backend();weights={k:np.zeros(v,np.float32) for k,v in m.WEIGHT_SHAPES.items()}
    # A simple fixture: x normal is 0.1 times red, y normal 0.1 times green.
    for layer in range(1,4):
        weights[f'fc{layer}_weight'][0,0]=1;weights[f'fc{layer}_weight'][1,1]=1
    weights['fc4_weight'][0,0]=.1;weights['fc4_weight'][1,1]=.1
    options={'input_range':'0_255','image_shape':[8,10],'model_channel_order':'RGB',
             'xy_mode':'normalized_row_col','zero_anchor':'mean','indentation_sign':1,'output_scale':1}
    return m,weights,options


def test_float_xy_and_explicit_channel_order():
    m=backend();image=np.zeros((3,4,3),np.uint8);image[...,0]=255
    f=m.features(image,model_channel_order='RGB',xy_mode='normalized_row_col').reshape(3,4,5)
    np.testing.assert_allclose(f[2,3],[1,0,0,2/3,3/4])
    bgr=m.features(image,model_channel_order='BGR',xy_mode='legacy_zero')
    np.testing.assert_allclose(bgr[0],[0,0,1,0,0])


def test_neumann_integrates_known_height_with_only_constant_ambiguity():
    m=backend();y,x=np.indices((31,43));height=.3*x-.2*y+np.sin(x/4)*np.cos(y/7)
    gx=np.zeros_like(height);gy=np.zeros_like(height)
    gx[:,:-1]=np.diff(height,axis=1);gy[:-1]=np.diff(height,axis=0)
    np.testing.assert_allclose(m.integrate_gradients(gx,gy),height-height.mean(),atol=2e-6)


def test_numeric_checkpoint_fixture_and_identical_reference_zero():
    m,weights,options=setup_backend();image=np.full((8,10,3),127,np.uint8)
    normals=m.infer_normals(image,weights,options)
    np.testing.assert_allclose(normals,127/255*.1,atol=1e-7)
    np.testing.assert_array_equal(m.reconstruct(image=image,baseline=image,calibration=weights,options=options),np.zeros((8,10)))


def test_fixed_sign_scale_and_anchor_are_required():
    m,weights,options=setup_backend();baseline=np.full((8,10,3),100,np.uint8);image=baseline.copy();image[:,4:,0]=200
    positive,_=m.signed_difference(image,baseline,weights,options)
    negative,_=m.signed_difference(image,baseline,weights,{**options,'indentation_sign':-1,'output_scale':2})
    np.testing.assert_allclose(negative,-2*positive)
    assert m.reconstruct(image=image,baseline=baseline,calibration=weights,options=options).min()==0
    with pytest.raises(ValueError,match='zero_anchor'):
        m.reconstruct(image=image,baseline=baseline,calibration=weights,options={**options,'zero_anchor':None})


def test_invalid_normals_and_unregistered_reference_rejected():
    m,weights,options=setup_backend();image=np.full((8,10,3),127,np.uint8)
    with pytest.raises(ValueError,match='registered reference'):
        m.reconstruct(image=image,baseline=image[:4],calibration=weights,options=options)
    weights['fc4_bias'][:]=2
    with pytest.raises(ValueError,match='Too many'):
        m.reconstruct(image=image,baseline=image,calibration=weights,options=options)


def test_marker_holes_do_not_become_false_zero_pressure_pixels():
    m,weights,options=setup_backend();image=np.full((8,10,3),150,np.uint8);image[3,4]=0
    height,stats=m.surface(image,weights,{**options,'marker_gray_threshold':70})
    clean,_=m.surface(np.full_like(image,150),weights,{**options,'marker_gray_threshold':70})
    assert stats['marker_fraction']==1/80
    np.testing.assert_allclose(height,clean)
