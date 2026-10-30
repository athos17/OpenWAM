"""Give the copied subset files their bdy modification times (some readers bind st_mtime_ns).

SharpaDex history manifests record each video's size and mtime_ns and refuse files that differ, so
a plain copy is rejected. Run on a800_1 (it reads the times from bdy over ssh):

    python sync_mtimes.py <subset root>/.copy_state/copy_list.tsv <subset root>
Files not copied yet are skipped; rerun after the copy completes (prepare_data.sh does).
"""
import os
import subprocess
import sys
from pathlib import Path

SRC_ROOT = "/mnt/pfs/ls3r34/workspace/dataset"
REMOTE = ("import os,sys\nroot=sys.argv[1]\nfor line in sys.stdin:\n    rel=line.rstrip('\\n')\n"
          "    print(rel+'\\t'+str(os.stat(os.path.join(root,rel)).st_mtime_ns))\n")


def main():
    copy_list, root = sys.argv[1], Path(sys.argv[2])
    rels = [line.split("\t")[2] for line in open(copy_list) if line.split("\t")[1] == "f"]
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "bdy_cpu",
                          f"/mnt/pfs/v89atp/workspace/envs/openwam/bin/python -c \"$(echo {REMOTE.encode().hex()} | xxd -r -p)\" {SRC_ROOT}"],
                         input="\n".join(rels) + "\n", capture_output=True, text=True, check=True).stdout
    done = skipped = 0
    for line in out.splitlines():
        rel, mtime = line.rsplit("\t", 1)
        path = root / rel
        if not path.is_file():
            skipped += 1
            continue
        st = path.stat()
        if st.st_mtime_ns != int(mtime):
            os.utime(path, ns=(st.st_atime_ns, int(mtime)))
        done += 1
    print(f"mtimes synced for {done} files, {skipped} not copied yet")


if __name__ == "__main__":
    main()
