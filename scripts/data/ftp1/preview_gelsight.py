#!/usr/bin/env python3
"""Small, explicitly UNVERIFIED Mini reconstruction experiments.

``sample`` reads only requested Zarr chunks from the existing FTP archives.
``render`` uses external public weights and creates inspectable comparisons.
Neither command writes a training sidecar, dataset publication or validated
calibration profile. Episode-first references are candidates, never certified
zero-contact measurements. No dataset is re-converted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def selected_frames(stream,spec,indices):
    """Decode only selected chunks; never instantiate a full-source array."""
    from convert import Array
    a=Array(stream,spec);indices=np.asarray(indices,dtype=np.int64)
    if (indices<0).any() or (indices>=a.shape[0]).any():raise ValueError('Sample index outside source array')
    if np.prod(a.chunk_shape,dtype=np.int64)*a.dtype.itemsize>128*1024**2:
        raise ValueError('Source chunk exceeds the experiment 128 MiB memory bound')
    out=np.empty((len(indices),)+a.shape[1:],dtype=a.dtype)
    for block in np.unique(indices//a.chunk_shape[0]):
        key='.'.join([str(int(block))]+['0']*(len(a.shape)-1))
        if key not in spec['chunks']:raise ValueError(f'Requested source sample is missing: {key}')
        offset,size=spec['chunks'][key];stream.seek(offset);raw=stream.read(size)
        if len(raw)!=size:raise EOFError('Short sampled source chunk')
        raw=a.codec.decode(raw) if a.codec else raw
        chunk=np.frombuffer(raw,dtype=a.dtype).reshape(a.chunk_shape,order=a.cfg.get('order','C'))
        positions=np.where(indices//a.chunk_shape[0]==block)[0]
        out[positions]=chunk[indices[positions]%a.chunk_shape[0]]
    return out


def sample(args):
    from convert import Parts,index_group
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    census=json.loads(Path(args.census).read_text());records=[]
    for source in args.sources:
        entry=next(x for x in census if x['dataset']==source)
        group=next(g for g in entry['groups'] if g['group']==args.group) if args.group else entry['groups'][0]
        out=root/source;out.mkdir(exist_ok=True)
        files=sorted(p for p in (Path(args.source_root)/source).iterdir() if '.tar' in p.name)
        index_path=Path(args.existing_index) if args.existing_index else out/'archive_index.json'
        if args.existing_index and not index_path.is_file():raise FileNotFoundError('Requested existing index not present')
        index=index_group(files,group,index_path)
        if index['arrays']['data/camera_main_rgb']['config']['shape'][0]!=group['frames']:raise ValueError('Existing index task length differs')
        if any(offset<group['offset'] or offset+size>group['end_offset']
               for spec in index['arrays'].values() for offset,size in spec['chunks'].values()):
            raise ValueError('Existing index contains chunks outside the requested task range')
        ends=group['episode_ends'];starts=[0]+ends[:-1];indices=[];episodes=[]
        for ep in range(min(args.episodes,len(ends))):
            local=np.unique(np.linspace(0,ends[ep]-starts[ep]-1,args.frames,dtype=int))
            positions=list(range(len(indices),len(indices)+len(local)))
            indices.extend((local+starts[ep]).tolist())
            episodes.append({'episode_index':ep,'source_start':starts[ep],'source_end':ends[ep],
                             'sample_positions':positions,'relative_frame_indices':local.tolist(),
                             'reference_sample_position':positions[0],'reference_status':'unverified_episode_first_frame'})
        values={};identities={}
        with Parts(files) as stream:
            for key,spec in index['arrays'].items():
                if key in ('data/camera_main_rgb','data/left_tactile_data_gripper','data/right_tactile_data_gripper',
                           'data/left_hand_joints','data/right_hand_joints'):
                    values[key.replace('/','__')]=selected_frames(stream,spec,indices)
                elif 'tactile_sensor_gripper' in key or 'tactile_type_gripper' in key:
                    identities[key]=selected_frames(stream,spec,[indices[0]]).tolist()
        np.savez_compressed(out/'samples.npz',**values)
        record={'source':source,'group':group['group'],'archive_fingerprint':index['archive_fingerprint'],
                'sample_file':'samples.npz','sample_sha256':digest(out/'samples.npz'),
                'absolute_source_frame_indices':indices,'episodes':episodes,'sensor_identity':identities,
                'array_shapes':{k:list(v.shape) for k,v in values.items()},'status':'unverified_preview_only'}
        save_json(out/'samples.json',record);records.append(record)
        print(json.dumps({'source':source,'sample_frames':len(indices),'shapes':record['array_shapes']}),flush=True)
    save_json(root/'sampling_summary.json',{'status':'unverified_preview_only','sources':records,
        'max_decoded_source_chunk_bytes':128*1024**2,'full_array_loading':False,'source_data_modified':False})


def numeric_weights(assets,output):
    import torch
    from depth_plugins.gelsight_mini import WEIGHT_SHAPES
    checkpoint=assets/'models__nnmini.pt';provenance=json.loads((assets/'provenance.json').read_text())
    if digest(checkpoint)!=provenance['assets']['models/nnmini.pt']['sha256']:raise ValueError('Public checkpoint hash changed')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict']
    out=output/'numeric_weights';out.mkdir(exist_ok=True);weights={};records={}
    for key,shape in WEIGHT_SHAPES.items():
        value=state[key.replace('_','.')].detach().cpu().numpy().astype(np.float32)
        if value.shape!=shape:raise ValueError('Unexpected public checkpoint architecture')
        p=out/f'{key}.npy';np.save(p,value,allow_pickle=False);weights[key]=value
        records[key]={'path':str(p),'sha256':digest(p),'shape':list(shape)}
    save_json(out/'provenance.json',{'source_checkpoint_sha256':digest(checkpoint),'source':provenance,
                                   'license':'GPL-3.0 repository license; no separate weight license found','arrays':records})
    return weights,provenance


def render(args):
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from depth_plugins import gelsight_mini as mini
    root=Path(args.samples);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assets=Path(args.assets);weights,provenance=numeric_weights(assets,out)
    options={'input_range':'0_255','image_shape':[240,320],'model_channel_order':'RGB',
             'xy_mode':'normalized_row_col','zero_anchor':'mean','indentation_sign':1,'output_scale':1,
             'marker_gray_threshold':70,'max_unreliable_fraction':.5}
    if not np.isfinite(args.preview_scale) or args.preview_scale<=0:raise ValueError('Positive fixed preview scale required')
    statistics=[];figures=[];diagnostic_arrays={}
    for source in args.sources:
        record=json.loads((root/source/'samples.json').read_text())
        if digest(root/source/'samples.npz')!=record['sample_sha256']:raise ValueError('Sample file fingerprint changed')
        with np.load(root/source/'samples.npz',allow_pickle=False) as data:
            for key in [k for k in data.files if 'tactile_data_gripper' in k]:
                frames=data[key][:,0];side='left' if 'left_' in key else 'right'
                for ep in record['episodes']:
                    positions=ep['sample_positions'];reference=frames[ep['reference_sample_position']]
                    fig,axes=plt.subplots(len(positions),6,figsize=(17,2.6*len(positions)),layout='constrained')
                    fig.suptitle(f'{source} / {side} / episode {ep["episode_index"]} — UNVERIFIED PREVIEW\n'
                                 f'First sampled frame is only a reference candidate; fixed scale {args.preview_scale:g} reconstruction units; not pressure',fontsize=12)
                    for row,pos in enumerate(positions):
                        current=frames[pos];signed,stats=mini.signed_difference(current,reference,weights,options)
                        normalized=np.clip(signed/args.preview_scale,0,1);small=cv2.resize(normalized,(32,32),interpolation=cv2.INTER_AREA)
                        alt_xy,_=mini.signed_difference(current,reference,weights,{**options,'xy_mode':'legacy_zero'})
                        alt_bgr,_=mini.signed_difference(current,reference,weights,{**options,'model_channel_order':'BGR'})
                        item={'source':source,'sensor_side':side,'episode_index':ep['episode_index'],
                              'source_frame':record['absolute_source_frame_indices'][pos],
                              'relative_frame':ep['relative_frame_indices'][row],
                              'reference_status':ep['reference_status'],'rgb_mean_abs_difference':float(np.abs(current.astype(float)-reference).mean()),
                              'signed_min':float(signed.min()),'signed_max':float(signed.max()),
                              'signed_abs_p95':float(np.quantile(np.abs(signed),.95)),
                              'positive_fraction':float((signed>0).mean()),'normalized_saturation_fraction':float((normalized>=1).mean()),
                              'legacy_xy_depth_mae':float(np.mean(np.abs(signed-alt_xy))),
                              'bgr_depth_mae':float(np.mean(np.abs(signed-alt_bgr))),**stats}
                        statistics.append(item)
                        name=f'{source}_{side}_ep{ep["episode_index"]}_frame{item["relative_frame"]}'
                        diagnostic_arrays[name+'_signed']=signed;diagnostic_arrays[name+'_normalized32']=small
                        images=[data['data__camera_main_rgb'][pos],current,reference,signed,normalized,small]
                        titles=['Scene','Stored tactile RGB','Unverified reference','Signed height difference','Positive / fixed scale','32×32 preview']
                        for col,(a,title) in enumerate(zip(images,titles)):
                            ax=axes[row,col]
                            if col<3:ax.imshow(a)
                            elif col==3:ax.imshow(a,cmap='coolwarm',vmin=-args.preview_scale,vmax=args.preview_scale)
                            else:ax.imshow(a,cmap='inferno',vmin=0,vmax=1,interpolation='nearest')
                            ax.set_xticks([]);ax.set_yticks([])
                            if row==0:ax.set_title(title,fontsize=10)
                            if col==0:ax.set_ylabel(f'frame {item["relative_frame"]}',fontsize=9)
                        axes[row,3].set_xlabel(f'{signed.min():.2f} … {signed.max():.2f} (relative units)',fontsize=8)
                        axes[row,5].set_xlabel(f'marker {stats["current"]["marker_fraction"]:.1%}; sat {item["normalized_saturation_fraction"]:.1%}',fontsize=8)
                    path=out/f'{source}_{side}_episode_{ep["episode_index"]}.png';fig.savefig(path,dpi=120);plt.close(fig);figures.append(path.name)
                    print(json.dumps({'source':source,'side':side,'episode':ep['episode_index'],'figure':str(path)}),flush=True)
    np.savez_compressed(out/'preview_arrays.npz',**diagnostic_arrays)
    report={'status':'unverified_preview_only','training_enabled':False,'options':options,'fixed_preview_scale':args.preview_scale,
            'normalization_warning':'Display-only positive signed height / fixed scale; no pressure or mm calibration',
            'reference_warning':'Episode-first frame has NOT been verified as unloaded; mean-zero Neumann height also lacks a verified noncontact anchor',
            'source_provenance':provenance,'backend_sha256':digest(mini.__file__),'figures':figures,'measurements':statistics,
            'differences_from_reference_sdk':['float normalized row/H,column/W instead of integer-truncated coordinates',
                                            'forward-difference least-squares Neumann solver with zero DC frequency',
                                            'nearest valid gradient interpolation for markers instead of upstream griddata',
                                            'explicit fixed reference instead of first-50-frame automatic zeroing'],
            'confirmed':['stored 240x320 RGB-size input and sensor fields','public checkpoint dimensions and hash','finite reconstruction for sampled frames','identical-image/reference gives zero'],
            'unverified':['actual source RGB/BGR convention and calibration training input convention','optical FOV/crop correspondence','sensor/session baseline','normal direction under real indentation','mm-per-pixel and pressure relation','noncontact boundary anchor','cross-device generalization']}
    save_json(out/'experiment.json',report)
    html=['<!doctype html><html><meta charset="utf-8"><title>FTP Mini reconstruction preview</title><body style="max-width:1400px;margin:auto;font-family:sans-serif">',
          '<h1>FTP GelSight Mini — UNVERIFIED preview</h1><p>真实小样本；参考帧未确认空载；输出为相对几何重建单位，不是压力。未启用训练。所有图使用同一固定显示尺度。</p>']
    for name in figures:html.append(f'<h2>{name}</h2><img style="width:100%" src="{name}">')
    html.append('<p>数值、来源哈希与具体限制见 experiment.json。</p></body></html>');(out/'index.html').write_text('\n'.join(html))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('sample');s.add_argument('--source-root',required=True);s.add_argument('--census',required=True);s.add_argument('--output',required=True)
    s.add_argument('--episodes',type=int,default=2);s.add_argument('--frames',type=int,default=5)
    s.add_argument('--group',help='Exact census group; requires exactly one source')
    s.add_argument('--existing-index',help='Reuse a known matching archive index without rewriting it; requires --group and one source')
    r=sub.add_parser('render');r.add_argument('--samples',required=True);r.add_argument('--assets',required=True);r.add_argument('--output',required=True)
    r.add_argument('--preview-scale',type=float,default=5)
    for subparser in (s,r):subparser.add_argument('--sources',nargs='+',choices=['Unit','Unit_Bimanual','VLA_touch'],default=['Unit','Unit_Bimanual','VLA_touch'])
    a=p.parse_args()
    if a.command=='sample' and (a.episodes<1 or a.frames<2):p.error('Require at least one episode and two sampled frames')
    if a.command=='sample' and (a.group or a.existing_index) and (len(a.sources)!=1 or not a.group):p.error('--group/--existing-index require one source and exact --group')
    if a.command=='sample':sample(a)
    else:render(a)
