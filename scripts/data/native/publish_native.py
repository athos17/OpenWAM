#!/usr/bin/env python3
"""Publish existing native LeRobot v3 data with immutable hardlinks.

Source metadata and bytes are never edited. Symlinks pointing inside the source
are rewritten relative to the output so source bucket and statistics links do
not silently resolve back into an old release. Restart verifies every link.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def publish(source, output):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output==source or output.is_relative_to(source):raise ValueError('Output must be independent of source')
    output.mkdir(parents=True,exist_ok=True)
    n=0; total=0; links=[]
    for root,dirs,files in os.walk(source,followlinks=False):
        rp=Path(root); dst=output/rp.relative_to(source);dst.mkdir(parents=True,exist_ok=True)
        for name in list(dirs)+files:
            src=rp/name;out=dst/name
            if src.is_symlink():
                target=src.resolve(strict=True)
                mapped=output/target.relative_to(source) if target.is_relative_to(source) else target
                desired=os.path.relpath(mapped,out.parent)
                if out.is_symlink():
                    if os.readlink(out)!=desired:raise ValueError(f'Existing symlink differs: {out}')
                elif out.exists():raise ValueError(f'Existing non-symlink: {out}')
                else:out.symlink_to(desired,target_is_directory=src.is_dir())
                links.append({'path':str(out.relative_to(output)),'target':desired})
                if name in dirs:dirs.remove(name)
            elif src.is_file():
                if out.exists():
                    if not os.path.samefile(src,out):raise ValueError(f'Existing file is not source hardlink: {out}')
                else:os.link(src,out)
                n+=1;total+=src.stat().st_size
    report={'schema':'touchwam_native_publication_v1','source':str(source),'output':str(output),
            'hardlinked_files':n,'referenced_bytes':total,'symlinks':links,
            'source_manifest_sha256':sha256(source/'dataset_manifest.json'),
            'policy':'Immutable files; never modify a hardlink in place. No resampling, selection, or reencoding.'}
    (output/'publication.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(publish(args.source,args.output)))
