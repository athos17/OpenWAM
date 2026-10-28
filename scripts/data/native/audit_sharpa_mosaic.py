from pathlib import Path
import pyarrow.parquet as pq,numpy as np,av,json,argparse
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
r=Path(args.dataset)
parts=sorted(r.glob('partitions/*/*'))
records=[]
for p in parts[::25]:
 ep=pq.read_table(next(p.glob('meta/episodes/*/*.parquet'))).to_pylist()[0]
 data=pq.read_table(p/'data'/f"chunk-{ep['data/chunk_index']:03d}"/f"file-{ep['data/file_index']:03d}.parquet",columns=['episode_index','observation.tactile']).to_pandas()
 forces=np.stack(data[data.episode_index==ep['episode_index']]['observation.tactile'])
 key='videos/observation.images.tactile_deform/'
 video=p/'videos/observation.images.tactile_deform'/f"chunk-{ep[key+'chunk_index']:03d}"/f"file-{ep[key+'file_index']:03d}.mp4"
 c=av.open(str(video));s=c.streams.video[0];s.thread_count=1
 off=float(ep[key+'from_timestamp']);c.seek(int(off/s.time_base),stream=s)
 vals=[];ff=[]
 for f in c.decode(s):
  idx=round((float(f.pts*s.time_base)-off)*30)
  if idx>=min(len(forces),1800):break
  if idx<0 or idx%15:continue
  a=f.to_ndarray(format='gray')
  means=np.array([a[y*240:(y+1)*240,x*240:(x+1)*240].mean() for y in range(2) for x in range(5)])
  vals.append(means);ff.append(np.abs(forces[idx].reshape(10,6)[:,:3]).sum(axis=1))
 c.close()
 if len(vals)>3:records.append((np.array(vals),np.array(ff)))
 print(p.name,len(vals),flush=True)
V=np.concatenate([x[0] for x in records]);F=np.concatenate([x[1] for x in records]);C=np.corrcoef(V.T,F.T)[:10,10:]
print('CORR',C.round(3));print('MAPPING',C.argmax(axis=1));print('SAMPLES',len(V))
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
np.savez(out/'sharpa_mapping_evidence.npz',tile_intensity=V,force_magnitude=F,correlation=C)
(out/'sharpa_mapping_evidence.json').write_text(json.dumps({'source':str(r),'partitions':len(records),'samples':len(V),'tile_to_numeric_finger':C.argmax(axis=1).tolist(),'diagonal_correlations':np.diag(C).tolist(),'method':'Pearson correlation between per-tile mean grayscale and absolute xyz force sum; independent identity sanity check, not pressure calibration'},indent=2)+'\n')
