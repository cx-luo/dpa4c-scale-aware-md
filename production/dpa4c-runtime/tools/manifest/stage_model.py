#!/usr/bin/env python3
"""Stage verified runtime members of the immutable model, without compilation."""
import hashlib,json,pathlib,zipfile

def digest(data):return hashlib.sha256(data).hexdigest()
def prepare_model(stage,engine):
    archive=pathlib.Path(engine['model_file'])
    if digest(archive.read_bytes())!=engine['model_sha256']:raise ValueError('model archive SHA mismatch')
    target=pathlib.Path(engine['environment']['DPA_AOTI_MODEL_DIRECTORY'])
    library=pathlib.Path(engine['environment']['DPA_AOTI_MODEL_LIBRARY'])
    if target.resolve()!=stage.resolve()/'models/aoti-model' or library.parent!=target:raise ValueError('model package layout mismatch')
    target.mkdir(parents=True,exist_ok=True);members=[];wrappers=[];names=set()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name=info.filename
            if info.is_dir() or ('/data/aotinductor/model/' not in name and '/data/constants/' not in name):continue
            p=pathlib.PurePosixPath(name)
            if '..' in p.parts or p.is_absolute():raise ValueError('unsafe model archive member')
            if p.suffix in ['.cpp','.o']:continue
            if p.name in names:raise ValueError('ambiguous model package member '+p.name)
            names.add(p.name);data=z.read(info);out=target/p.name
            if out.exists():
                if digest(out.read_bytes())!=digest(data):raise ValueError('staged model member mismatch '+p.name)
            else:out.write_bytes(data)
            members.append({'path':str(out.relative_to(stage)),'archive_member':name,'sha256':digest(data),'source':'immutable compressed model archive','build_target':None,'provided_by_runtime_image':False})
            if p.name.endswith('.wrapper.so'):wrappers.append(out)
    if wrappers!=[library]:raise ValueError('model wrapper identity mismatch')
    manifest={'model_archive_sha256':engine['model_sha256'],'members':members,'runtime_compilation':False,'runtime_extraction':False}
    (stage/'manifest/model-package.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {r['path']:r for r in members}
