import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import torch
import time
from mmengine import Config
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


if __name__ == '__main__':
    input_res = (1920, 1080)
    config_path = r'local_configs\ddrnet\ddrnet_23-slim_in1k-pre_1xb12-40k_msd-832x832.py'
    register_all_modules()
    cfg = Config.fromfile(config_path)
    device = torch.device('cuda')
    model = MODELS.build(cfg.model)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, input_res[0], input_res[1]).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input, mode='predict')
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input, mode='predict')
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input, mode='predict')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    
    