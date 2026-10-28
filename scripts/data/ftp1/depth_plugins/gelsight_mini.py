"""Offline GelSight Mini normal-network backend with explicit calibration.

The MLP dimensions and inference feature convention follow gsrobotics commit
321d6a22da64529138ff10237335038fd8c5189f. The caller supplies numeric NPY weights,
not a pickled checkpoint, and a spatially registered reference RGB image.
Weights, sensor optics, reference, sign, boundary anchor and output scale must
be validated for a real sensor before this backend supplies training labels.

This implementation uses a forward-difference Neumann least-squares Poisson
solver with its correct zero DC mode. It intentionally does not reproduce the
upstream solver's eigenvalue indexing. Explicit ``legacy_zero`` XY is provided
for diagnostic comparison to upstream integer-coordinate truncation only.
No reference is estimated from the input sequence and no per-frame rescaling
is performed. Height/indentation is not pressure.
"""
from __future__ import annotations

import numpy as np
from scipy import fft, ndimage


WEIGHT_SHAPES = {
    'fc1_weight': (64, 5), 'fc1_bias': (64,),
    'fc2_weight': (64, 64), 'fc2_bias': (64,),
    'fc3_weight': (64, 64), 'fc3_bias': (64,),
    'fc4_weight': (2, 64), 'fc4_bias': (2,),
}


def validate_image(image):
    image=np.asarray(image,dtype=np.float32)
    if image.ndim!=3 or image.shape[-1]!=3 or min(image.shape[:2])<3:
        raise ValueError('Expected registered H x W x 3 camera image')
    if not np.isfinite(image).all() or image.min()<0 or image.max()>255:
        raise ValueError('Camera RGB must be finite and in the declared 0..255 range')
    return image


def features(image, *, model_channel_order, xy_mode):
    """Return RGB/255 followed by float row/H, column/W (not x/y swapped)."""
    image=validate_image(image);h,w=image.shape[:2]
    if model_channel_order not in ('RGB','BGR'):
        raise ValueError('Explicit model_channel_order RGB or BGR is required')
    rgb=image if model_channel_order=='RGB' else image[...,::-1]
    xy=np.indices((h,w),dtype=np.float32).transpose(1,2,0)
    if xy_mode=='normalized_row_col':xy/=np.array([h,w],np.float32)
    elif xy_mode=='legacy_zero':xy[:]=0
    else:raise ValueError('Explicit normalized_row_col or diagnostic legacy_zero XY mode is required')
    return np.concatenate((rgb/255,xy),axis=-1).reshape(-1,5)


def integrate_gradients(gx,gy):
    """Least-squares height for forward x=columns/y=rows edge gradients.

    Solve D.T D h = D.T g on the rectangular pixel graph. The unobservable
    additive constant is fixed to mean zero. Last-column gx and last-row gy
    have no forward edge and are ignored.
    """
    gx=np.asarray(gx,dtype=np.float64);gy=np.asarray(gy,dtype=np.float64)
    if gx.ndim!=2 or gx.shape!=gy.shape or min(gx.shape)<2 or not np.isfinite(gx).all() or not np.isfinite(gy).all():
        raise ValueError('Finite matching 2D gradient maps are required')
    rhs=np.zeros_like(gx)
    rhs[:,:-1]-=gx[:,:-1];rhs[:,1:]+=gx[:,:-1]
    rhs[:-1,:]-=gy[:-1,:];rhs[1:,:]+=gy[:-1,:]
    h,w=gx.shape
    eigen=(2-2*np.cos(np.pi*np.arange(h)/h))[:,None]+(2-2*np.cos(np.pi*np.arange(w)/w))[None,:]
    eigen[0,0]=1
    spectrum=fft.dctn(rhs,type=2,norm='ortho')/eigen;spectrum[0,0]=0
    return fft.idctn(spectrum,type=2,norm='ortho').astype(np.float32)


def infer_normals(image,calibration,options):
    image=validate_image(image)
    if options.get('input_range')!='0_255':
        raise ValueError('input_range must explicitly declare 0_255')
    if list(image.shape[:2])!=options.get('image_shape'):
        raise ValueError('Image shape must match the calibrated image_shape')
    weights={}
    for key,shape in WEIGHT_SHAPES.items():
        value=np.asarray(calibration[key],dtype=np.float32)
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid network calibration tensor {key}')
        weights[key]=value
    inputs=features(image,model_channel_order=options.get('model_channel_order'),xy_mode=options.get('xy_mode'))
    block=options.get('inference_chunk_pixels',8192)
    if type(block) is not int or block<1:raise ValueError('Positive inference_chunk_pixels required')
    output=np.empty((len(inputs),2),np.float32)
    for start in range(0,len(inputs),block):
        value=inputs[start:start+block]
        for layer in range(1,5):
            value=value@weights[f'fc{layer}_weight'].T+weights[f'fc{layer}_bias']
            if layer<4:np.maximum(value,0,out=value)
        output[start:start+len(value)]=value
    return output.reshape(*image.shape[:2],2)


def surface(image,calibration,options):
    """Return centered reconstructed surface and explicit numerical diagnostics."""
    image=validate_image(image);normals=infer_normals(image,calibration,options)
    norm2=np.sum(normals**2,axis=-1)
    bad_normals=(~np.isfinite(norm2))|(norm2>=1-1e-6)
    threshold=options.get('marker_gray_threshold')
    if threshold is None:marker=np.zeros(image.shape[:2],bool)
    else:
        if not np.isfinite(threshold) or not 0<=threshold<=255:raise ValueError('Invalid fixed marker threshold')
        marker=image@np.array([.299,.587,.114],np.float32)<=threshold
    unreliable=bad_normals|marker
    max_fraction=float(options.get('max_unreliable_fraction',.5))
    if not 0<=max_fraction<1 or unreliable.mean()>max_fraction:
        raise ValueError('Too many masked/invalid pixels for reliable normal interpolation')
    nz=np.sqrt(np.maximum(1-norm2,1e-6))
    gradients=-normals/nz[...,None]
    if unreliable.any():
        # Nearest measured neighbour is deterministic; this is an explicit
        # alternative to upstream griddata, not a force estimate at markers.
        indices=ndimage.distance_transform_edt(unreliable,return_distances=False,return_indices=True)
        gradients[unreliable]=gradients[indices[0][unreliable],indices[1][unreliable]]
    height=integrate_gradients(gradients[...,0],gradients[...,1])
    return height,{'invalid_normal_fraction':float(bad_normals.mean()),'marker_fraction':float(marker.mean()),
                   'unreliable_fraction':float(unreliable.mean()),'normal_norm2_max':float(norm2.max()),
                   'gradient_abs_p99':float(np.quantile(np.abs(gradients),.99))}


def signed_difference(image,baseline,calibration,options):
    baseline=validate_image(baseline);image=validate_image(image)
    if baseline.shape!=image.shape:raise ValueError('A spatially registered reference RGB image is required')
    current,current_stats=surface(image,calibration,options)
    reference,reference_stats=surface(baseline,calibration,options)
    value=current-reference
    anchor=options.get('zero_anchor')
    if anchor=='mask_median':
        mask=np.asarray(calibration['zero_anchor_mask'])
        if mask.shape!=value.shape or not np.isfinite(mask).all() or not ((mask==0)|(mask==1)).all() or not mask.any():
            raise ValueError('zero_anchor_mask must be a nonempty registered binary mask')
        value-=np.median(value[mask.astype(bool)])
    elif anchor!='mean':raise ValueError('Explicit zero_anchor mean or mask_median is required')
    sign=options.get('indentation_sign');scale=float(options.get('output_scale',0))
    if sign not in (-1,1) or not np.isfinite(scale) or scale<=0:raise ValueError('Calibrated indentation_sign and positive output_scale are required')
    return (value*sign*scale).astype(np.float32),{'current':current_stats,'reference':reference_stats}


def reconstruct(*,image,baseline,calibration,options):
    """Plugin API: positive relative indentation; caller performs fixed scaling."""
    value,_=signed_difference(image,baseline,calibration,options)
    return np.maximum(value,0).astype(np.float32)
