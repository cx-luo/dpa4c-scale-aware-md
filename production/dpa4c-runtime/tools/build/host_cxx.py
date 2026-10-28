#!/usr/bin/env python3
"""Fix the pinned PPU SDK's generated host parameter declarator, then run GCC."""
import hashlib,json,os,pathlib,re,sys
NVCC_SHA='45cbe7094fa2d2b046f793ced069819eb1111e2df1f93ef00e6b48d77fd7623a'
CXX_SHA='1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769'
def sha(data):return hashlib.sha256(data).hexdigest()
def repair(text):
    changes=[];lines=text.splitlines(keepends=True)
    for i,line in enumerate(lines):
        if '__device_stub__' not in line and '__wrapper__device_stub_' not in line:continue
        brace=line.find('{')
        if brace<0:continue
        signature=line[:brace].rstrip()
        if not signature.endswith(')'):continue
        depth=1;pos=len(signature)-2
        while pos>=0:
            if signature[pos]==')':depth+=1
            elif signature[pos]=='(':
                depth-=1
                if depth==0:break
            pos-=1
        if pos<0:raise RuntimeError('generated kernel declaration is unbalanced')
        parameter=signature[pos+1:-1]
        if not re.search(r'const\s+::Kokkos::Impl::Parallel(?:For|Scan|Reduce)<',parameter):continue
        if re.search(r'\bdriver\s*$',parameter):continue
        if re.search(r'\bdriver\b',parameter):
            if len(re.findall(r'\bdriver\b',parameter))!=1:raise RuntimeError('unexpected driver token count in generated host declaration')
            if not re.search(r'\*\s*driver\s*(?:\[|\))',parameter):raise RuntimeError('unrecognized generated parameter corruption')
            repaired=re.sub(r'\bdriver\b','',parameter)+' driver'
            fixed=signature[:pos+1]+repaired+')'
            new=fixed+line[len(signature):]
            changes.append({'line':i+1,'before_signature_sha256':sha(signature.encode()),'after_signature_sha256':sha(fixed.encode()),'role':'move parameter name out of nested array/member-pointer type to the outer parameter'})
            lines[i]=new
    return ''.join(lines),changes
def main():
    cxx=pathlib.Path('/usr/bin/g++-13')
    if sha(cxx.read_bytes())!=CXX_SHA:raise RuntimeError('host compiler hash mismatch')
    nvcc=pathlib.Path(os.environ.get('CUDA_HOME','/usr/local/PPU_SDK/CUDA_SDK'))/'bin/nvcc'
    if sha(nvcc.read_bytes())!=NVCC_SHA:raise RuntimeError('PPU SDK compiler hash mismatch')
    inputs=[]
    for arg in sys.argv[1:]:
        p=pathlib.Path(arg)
        if not p.is_file() or p.suffix not in ['.cpp','.o'] or '-c' not in sys.argv:continue
        tmp=pathlib.Path(os.environ.get('TMPDIR','/workspace')).resolve()
        if not p.resolve().is_relative_to(tmp):continue
        data=p.read_bytes()
        if data[:4]==b'\x7fELF':continue
        try:text=data.decode()
        except UnicodeError:continue
        if '__hggc' not in text:continue
        for include in re.findall(r'^#include\s+"([^"\n]*-host-[^"\n]*\.cpp)"',text,re.M):
            child=(p.parent/include).resolve()
            if not child.is_relative_to(tmp) or not child.is_file():raise RuntimeError('generated host include outside owned temporary directory')
            inputs.append(child)
        if '__wrapper__device_stub_' in text:inputs.append(p)
    for p in set(inputs):
        data=p.read_bytes();text=data.decode()
        transformed,changes=repair(text)
        if not changes:continue
        p.write_text(transformed,newline='\n')
        log=pathlib.Path(os.environ['TMPDIR'])/'host-source-fixes.jsonl'
        record={'path':str(p),'input_sha256':sha(data),'output_sha256':sha(p.read_bytes()),'compiler_sha256':NVCC_SHA,'changes':changes}
        fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
        try:os.write(fd,(json.dumps(record,separators=(',',':'))+'\n').encode())
        finally:os.close(fd)
    os.execv(str(cxx),[str(cxx),*sys.argv[1:]])
if __name__=='__main__':main()
