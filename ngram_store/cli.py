"""NGramStore: pack, inspect and query frozen n-gram vector tables."""
import argparse
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

from . import NativeRows, pack_native

FORWARD={
    'prepare-fasttext':('prepare_fasttext_corpus','Prepare plain text for the custom FastText SGNS trainer.'),
    'fasttext-export':('fasttext_native_bundle','Export a dense legacy FastText model into the same native backend.'),
    'tiny-export':('tinyengram_checkpoint','Export a TinyEngram/Qwen checkpoint.'),
    'tiny-verify':('verify_tinyengram_checkpoint','Verify a converted TinyEngram checkpoint on CPU.'),
}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):digest.update(part)
    return digest.hexdigest()


def pack_store(source, output, dim, *, native_library=None, keys=None, structural_library=None,
               mode='exact', cosine=.99, norm_ratio=.99, max_block_bytes=8192, deduplicate=False):
    if sys.byteorder!='little':raise ValueError('CLI raw input requires little-endian IEEE float32 host')
    source,output=Path(source).resolve(),Path(output).resolve()
    if type(dim) is not int or dim<=0:raise ValueError('dim must be positive')
    rows,tail=divmod(source.stat().st_size,dim*4)
    if not rows or tail:raise ValueError('input must contain nonempty complete float32 rows')
    if output.exists():raise FileExistsError('use a fresh store directory')
    if mode not in ('layout','exact','joint'):raise ValueError('invalid structural mode')
    if keys is None and mode=='joint':raise ValueError('joint sharing requires registered string keys')
    if keys is not None and deduplicate:raise ValueError('structural representatives and generic deduplication are separate modes')
    source_hash=sha256(source); output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ngram-store-',dir=output.parent) as tmp:
        candidate=Path(tmp)/'bundle'
        if keys is not None:
            module=import_module('meft_memory._runtime.structural_native')
            with module.StructuralPlan(source.read_bytes(),dim,keys,structural_library,
                                       mode=mode,cosine=cosine,norm_ratio=norm_ratio) as plan:
                storage=plan.export(source,candidate,max_block_bytes=max_block_bytes)
                expected=plan.reconstructed_sha256; preserved=plan.original_bytes_preserved
        else:
            candidate.mkdir()
            storage=pack_native(source,candidate/'table.men',dim*4,256,deduplicate,predictor='xor',
                                max_block_bytes=max_block_bytes)
            expected=source_hash; preserved=True
        reconstructed=hashlib.sha256()
        with NativeRows(candidate/'table.men',native_library) as table:
            for begin in range(0,rows,4096):reconstructed.update(table.rows(range(begin,min(begin+4096,rows))))
        if reconstructed.hexdigest()!=expected:raise ValueError('full-row reconstruction differs from selected representatives')
        if preserved and expected!=source_hash:raise ValueError('lossless pack changed source bytes')
        if sha256(source)!=source_hash:raise ValueError('source changed during export')
        artifact_hash=sha256(candidate/'table.men')
        # The same artifact can be opened by the existing frozen PyTorch adapter.
        embedding={'schema_version':1,'table_file':'table.men','shape':[rows,dim],
            'dtype':'torch.float32','byteorder':'little','head_sizes':[rows],'offsets':[0],
            'backend':'native','weight_sha256':expected,'source_weight_sha256':source_hash,
            'artifact_sha256':artifact_hash,'approximate':not preserved,**storage}
        (candidate/'embedding.json').write_text(json.dumps(embedding,indent=2)+'\n')
        metadata={'schema_version':1,'engine':'NGramStore','shape':[rows,dim],'dtype':'float32-le',
            'source_sha256':source_hash,'reconstructed_sha256':expected,'artifact_sha256':artifact_hash,
            'original_bytes_preserved':preserved,'keyed_rows':len(keys) if keys is not None else 0,
            'structural_mode':mode if keys is not None else None,'storage':storage,
            'contract':'Original numeric IDs; decoder preserves representative bytes. Joint sharing can change source weights.',
            'limitations':['Frozen single-worker handles; cache budget is not an RSS bound.',
                           'Structural mode requires trusted Linux x86-64 DATrie artifacts and full offline planning.',
                           'Hash collisions do not certify prefix/suffix relations.']}
        (candidate/'store.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if output.exists():raise FileExistsError('store directory appeared during export')
        os.rename(candidate,output)
    return metadata


def read_keys(path):
    raw=json.loads(Path(path).read_text())
    if not isinstance(raw,dict):raise ValueError('keys JSON must map original row IDs to UTF-8 strings')
    result={}
    for key,value in raw.items():
        if not key.isdigit() or str(int(key))!=key or not isinstance(value,str):
            raise ValueError('keys must use canonical nonnegative row IDs and string values')
        result[int(key)]=value
    return result


def open_metadata(path):
    root=Path(path)
    name=next((name for name in ('store.json','fasttext.json','manifest.json') if (root/name).is_file()),None)
    if name is None:raise ValueError('store metadata not found')
    metadata=json.loads((root/name).read_text())
    if metadata.get('schema_version')!=1:raise ValueError('unsupported store metadata')
    return metadata


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in FORWARD:
        name=argv[0]; old=sys.argv
        try:
            sys.argv=['ngram-store '+name,*argv[1:]]
            return import_module('meft_memory._runtime.'+FORWARD[name][0]).main()
        finally:sys.argv=old
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    pack=commands.add_parser('pack',help='Pack existing little-endian float32 rows and verify every reconstructed row.')
    pack.add_argument('--input',type=Path,required=True); pack.add_argument('--output',type=Path,required=True)
    pack.add_argument('--dim',type=int,required=True); pack.add_argument('--native-library',type=Path)
    pack.add_argument('--keys',type=Path); pack.add_argument('--structural-library',type=Path)
    pack.add_argument('--mode',choices=('layout','exact','joint'),default='exact')
    pack.add_argument('--cosine',type=float,default=.99); pack.add_argument('--norm-ratio',type=float,default=.99)
    pack.add_argument('--max-block-bytes',type=int,default=8192); pack.add_argument('--deduplicate',action='store_true')
    compact=commands.add_parser('compact-bags',help='Build exact incidence groups and FP32 group means with disk-backed buffers.')
    compact.add_argument('--input',type=Path,required=True); compact.add_argument('--dim',type=int,required=True)
    compact.add_argument('--bag-ids',type=Path,required=True); compact.add_argument('--bag-offsets',type=Path,required=True)
    compact.add_argument('--protected-ids',type=Path,help='Raw uint64 independently addressed IDs that must remain singletons.')
    compact.add_argument('--output',type=Path,required=True); compact.add_argument('--memory-mib',type=int,default=64)
    compact.add_argument('--native',action='store_true',help='Produce a ready-to-query certified native bundle; omit intermediate means/maps.')
    compact.add_argument('--block-rows',type=int,default=8,help='Raw native block width for --native (default: 8).')
    query_bags=commands.add_parser('query-bags',help='Certify and evaluate original-ID bags on a compact native bundle.')
    query_bags.add_argument('--store',type=Path,required=True)
    query_bags.add_argument('--native-library',type=Path)
    bag_selection=query_bags.add_mutually_exclusive_group(required=True)
    bag_selection.add_argument('--ids',type=int,nargs='+',help='One bag, including repeated original IDs.')
    bag_selection.add_argument('--bags',type=Path,help='JSON list of bags (lists of original row IDs), including empty bags.')
    query_bags.add_argument('--mode',choices=('sum','mean'),default='sum')
    query_bags.add_argument('--bounds',type=Path,help='Optional actual-source residual sidecar; requires an explicit error budget.')
    query_bags.add_argument('--max-relative-l2',type=float,help='Maximum source-relative L2 upper bound (0.01 = 1%%).')
    query_bags.add_argument('--max-absolute-l2',type=float,help='Maximum absolute L2 upper bound; both supplied budgets must pass.')
    bounds=commands.add_parser('build-error-bounds',help='Build actual-source residual bounds for opt-in partial-group queries.')
    bounds.add_argument('--input',type=Path,required=True)
    bounds.add_argument('--store',type=Path,required=True)
    bounds.add_argument('--output',type=Path,required=True)
    bounds.add_argument('--native-library',type=Path)
    bounds.add_argument('--chunk-rows',type=int,default=1024)
    demo_bags=commands.add_parser('demo-bags',help='Train an offline NumPy example and verify native certified queries.')
    demo_bags.add_argument('--output',type=Path,required=True)
    demo_bags.add_argument('--native-library',type=Path)
    inspect=commands.add_parser('inspect',help='Inspect row shape, storage bytes and fidelity contract.')
    query=commands.add_parser('query',help='Query original row IDs or registered string keys.')
    for command in (inspect,query):
        command.add_argument('--store',type=Path,required=True); command.add_argument('--native-library',type=Path)
    selection=query.add_mutually_exclusive_group(required=True)
    selection.add_argument('--ids',type=int,nargs='+'); selection.add_argument('--keys',nargs='+')
    query.add_argument('--structural-library',type=Path); query.add_argument('--cache-bytes',type=int,default=4*1024**2)
    query.add_argument('--hex',action='store_true',help='Return original row bytes, including nonfinite/signed-zero bits.')
    for name,(_,help_text) in FORWARD.items():commands.add_parser(name,help=help_text,add_help=False)
    args=parser.parse_args(argv)
    try:
        if args.command=='build-error-bounds':
            from . import build_error_bounds
            result=build_error_bounds(args.input,args.store,args.output,
                native_library=args.native_library,chunk_rows=args.chunk_rows)
        elif args.command=='compact-bags':
            if args.native:
                module=import_module('meft_memory._runtime.certified_store')
                result=module.compact_native(args.input,args.dim,args.bag_ids,args.bag_offsets,args.output,
                    memory_bytes=args.memory_mib*1024**2,protected_ids=args.protected_ids,block_rows=args.block_rows)
                print(json.dumps(result,indent=2,allow_nan=False))
                return
            module=import_module('meft_memory._runtime.external_cotraining')
            if args.dim<=0:raise ValueError('dim must be positive')
            rows,tail=divmod(args.input.stat().st_size,args.dim*4)
            if not rows or tail:raise ValueError('complete nonempty FP32 rows required')
            output=args.output.resolve()
            output.mkdir(parents=True,exist_ok=False)
            try:
                partition=module.fit_external(module.iter_bag_files(args.bag_ids,args.bag_offsets),
                    rows,output/'partition',memory_bytes=args.memory_mib*1024**2,
                    protected=module.iter_protected_ids(args.protected_ids) if args.protected_ids else ())
                projected=partition.project_file(args.input,args.dim,output/'means.f32',
                    memory_bytes=args.memory_mib*1024**2)
                partition.save_npz(output/'domain.npz')
                result={'partition':partition.summary(),'projection':projected,
                    'source_row_bytes_preserved':False,'mean_row_ids':'Group IDs from domain.npz mapping.',
                    'contract':'Complete certified bag sums/means in real arithmetic; reject incomplete groups.',
                    'limitations':['Managed buffers are not a hard RSS or filesystem page-cache cap.',
                        'Runtime map loading and native packing have separate memory costs.',
                        'Native numeric lookup alone does not enforce the bag certificate.']}
                (output/'compact.json').write_text(json.dumps(result,indent=2)+'\n')
            except BaseException:
                shutil.rmtree(output)
                raise
        elif args.command=='demo-bags':
            module=import_module('meft_memory._runtime.demo_certified_store')
            result=module.demo(args.output,native_library=args.native_library)
        elif args.command=='query-bags':
            bags=[args.ids] if args.ids is not None else json.loads(args.bags.read_text())
            if not isinstance(bags,list) or any(not isinstance(bag,list) for bag in bags):
                raise ValueError('JSON list of original-ID bags required')
            budget=args.max_relative_l2 is not None or args.max_absolute_l2 is not None
            if (args.bounds is not None) != budget:
                raise ValueError('--bounds and an explicit L2 error budget must be supplied together')
            if budget:
                from . import BudgetedBagRows
                with BudgetedBagRows(args.store,args.bounds,args.native_library) as table:
                    assessment=table.aggregate(bags,mean=args.mode=='mean',
                        max_relative_l2=args.max_relative_l2,max_absolute_l2=args.max_absolute_l2)
                    result={'mode':args.mode,'certified':False,'approximation_opt_in':True,
                        'source_row_bytes_preserved':False,'source_sha256':table.source_sha256,
                        'vectors':assessment.vectors.tolist(),'bounds':assessment.summary(),
                        'max_relative_l2':args.max_relative_l2,'max_absolute_l2':args.max_absolute_l2}
            else:
                from . import CertifiedBagRows
                with CertifiedBagRows(args.store,args.native_library) as table:
                    plan=table.compile(bags)
                    values=table.evaluate(plan,mean=args.mode=='mean')
                    result={'mode':args.mode,'certified':True,'source_row_bytes_preserved':False,
                        'vectors':values.tolist(),'plan':plan.summary(),
                        'contract':'Certified projected aggregates; floating-point rounding is separate.'}
        elif args.command=='pack':
            result=pack_store(args.input,args.output,args.dim,native_library=args.native_library,
                keys=read_keys(args.keys) if args.keys else None,structural_library=args.structural_library,
                mode=args.mode,cosine=args.cosine,norm_ratio=args.norm_ratio,
                max_block_bytes=args.max_block_bytes,deduplicate=args.deduplicate)
        else:
            metadata=open_metadata(args.store); rows,dim=metadata['shape']
            if args.command=='query' and metadata.get('name')=='certified-groups':
                raise ValueError('certified aggregate store: use query-bags to check the original-ID bag')
            ids=None
            if args.command=='query':
                ids=args.ids
                if args.keys is not None:
                    module=import_module('meft_memory._runtime.structural_native')
                    with module.StructuralIndex(args.store/'keys.das',args.structural_library) as index:ids=index.lookup(args.keys)
                    if None in ids:raise ValueError('query contains an unregistered key')
            with NativeRows(args.store/'table.men',args.native_library,
                            args.cache_bytes if args.command=='query' else 0) as table:
                if table.header['n_rows']!=rows or table.header['row_bytes']!=dim*4:
                    raise ValueError('row-store shape differs from metadata')
                if ids is None:result=metadata
                else:
                    raw=table.rows(ids)
                    result={'ids':ids,'original_bytes_preserved':metadata['original_bytes_preserved']}
                    if args.hex:result['rows_hex']=[raw[i*dim*4:(i+1)*dim*4].hex() for i in range(len(ids))]
                    else:result['vectors']=[list(struct.unpack_from('<'+str(dim)+'f',raw,i*dim*4)) for i in range(len(ids))]
        print(json.dumps(result,indent=2,allow_nan=False))
    except (ValueError,OSError,KeyError,TypeError) as exc:parser.error(str(exc))


if __name__=='__main__':main()
