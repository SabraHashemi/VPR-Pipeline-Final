import argparse, os, json, numpy as np, time
from utils.logging import info,warn
from retrieval.extractor import extract_descriptors
from retrieval.index_search import build_faiss_index, search_index
from metrics.retrieval_metrics import compute_all_recalls
from rerank.light_rerank import rerank_topk_by_inliers
from rerank.full_rerank import FullReranker
from logger.wandb_logger import WandBLogger
import torch
from models.resnet_backbone import build_resnet50
from models.netvlad_backbone import build_netvlad
from models.mixvpr_backbone import build_mixvpr
from utils.image_io import load_dataset_paths, list_images, load_rgb

def build_backbone(name='resnet50', pretrained=True, device='cpu'):
    name = name.lower()
    if name == 'resnet50': return build_resnet50(pretrained=pretrained, device=device)
    if name == 'netvlad': return build_netvlad(pretrained=pretrained, device=device)
    if name == 'mixvpr': return build_mixvpr(pretrained=pretrained, device=device)
    raise ValueError('Unknown backbone: '+str(name))

def run_pipeline(args):
    t0=time.time()
    os.makedirs('outputs', exist_ok=True); os.makedirs('matches', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    info(f'Using device: {device}')
    logger = WandBLogger(project='VPR_Benchmark', run_name=args.run_name, enabled=not args.no_wandb)
    model = build_backbone(name=args.backbone, pretrained=True, device=device)
    if args.query:
        db_imgs = list_images(args.db)
        q_imgs = list_images(args.query)
    else:
        db_imgs, q_imgs = load_dataset_paths(args.db)
    info(f'Extracting descriptors db={len(db_imgs)}, queries={len(q_imgs)}')
    db_desc, db_paths = extract_descriptors(model, db_imgs, batch_size=args.batch, device=device, size=args.size)
    q_desc, q_paths = extract_descriptors(model, q_imgs, batch_size=args.batch, device=device, size=args.size)
    index = build_faiss_index(db_desc)
    D, I, qtime = search_index(index, q_desc, k=args.k)
    gt = [next((i for i,p in enumerate(db_paths) if os.path.basename(p)==os.path.basename(q)),0) for q in q_paths]
    recalls_pre = compute_all_recalls(I, gt)
    info(f'Pre-rerank recalls: {recalls_pre}')
    mode = args.mode; full_reranker = None
    if mode == 'full':
        full_reranker = FullReranker(device=device)
        if not full_reranker.available():
            warn('Full reranker not available; falling back to light mode')
            mode = 'light'
    new_orders = []
    for qi, qpath in enumerate(q_paths):
        topk_idx = I[qi,:args.k]
        topk_paths = [db_paths[i] for i in topk_idx]
        if mode == 'light':
            order, scores = rerank_topk_by_inliers(qpath, topk_paths)
        else:
            order, scores = full_reranker.rerank(qpath, topk_paths)
        best_idx = topk_idx[order[0]] if len(order)>0 else topk_idx[0]
        from PIL import Image
        qimg = load_rgb(qpath); dimg = load_rgb(db_paths[best_idx])
        canvas = Image.new('RGB', (max(qimg.width,dimg.width), qimg.height + dimg.height))
        canvas.paste(qimg, (0,0)); canvas.paste(dimg, (0,qimg.height))
        canvas.save(os.path.join('matches', f'q{qi:04d}_best.jpg'))
        new_orders.append([topk_idx[i] for i in order])
    I2 = np.array(new_orders)
    recalls_post = compute_all_recalls(I2, gt)
    info(f'Post-rerank recalls: {recalls_post}')
    results = {'backbone': args.backbone, 'mode': mode, 'recall_pre': recalls_pre, 'recall_post': recalls_post, 'time_sec': time.time()-t0}
    with open('outputs/pipeline_results.json','w') as f: json.dump(results,f,indent=2)
    info('Saved outputs/pipeline_results.json')
    logger.log({'recall_pre': results['recall_pre'], 'recall_post': results['recall_post'], 'time_sec': results['time_sec']})
    logger.finish()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    pl = sub.add_parser('pipeline')
    pl.add_argument('--db', required=True)
    pl.add_argument('--query')
    pl.add_argument('--k', type=int, default=5)
    pl.add_argument('--mode', choices=['light','full'], default='light')
    pl.add_argument('--batch', type=int, default=8)
    pl.add_argument('--size', type=int, default=480)
    pl.add_argument('--use_gpu', action='store_true')
    pl.add_argument('--no_wandb', action='store_true')
    pl.add_argument('--run_name', type=str, default=None)
    pl.add_argument('--backbone', choices=['resnet50','netvlad','mixvpr'], default='resnet50')
    args = p.parse_args()
    if args.cmd == 'pipeline': run_pipeline(args)
    else: p.print_help()
