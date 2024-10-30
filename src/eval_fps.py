import re
import json
import torch
import subprocess
from tqdm import tqdm
from time import time
# from thop import profile
from lightning.pytorch.utilities.model_summary import ModelSummary

from config import cfg
from utils.misc import param_count, param_size

from models.handmvnet import HandMvNet as Model
from models.joints_to_vertices import JointsToVertices


torch.backends.cudnn.benchmark = True


class InferenceSpeedTest:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        cfg["train"]["device"] = self.device
        self.print_config()
        self.print_system_info()
        self.init_model()
        self.print_model_summary()
        self.loop()
    
    def print_model_summary(self):
        print("-------------------------------------------------")
        summary = ModelSummary(self.model, max_depth=2)
        print(summary)
        print(param_size(self.model), "MB")
        print(param_count(self.model), "M")
        print("-------------------------------------------------")

    def print_config(self):
        print("-------------------------------------------------")
        print("Config:", json.dumps(cfg, indent=2))
        
    def print_system_info(self):
        print("-------------------------------------------------")
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        cpu_info = " "
        for line in all_info.split("\n"):
            if "model name" in line:
                cpu_info = re.sub( ".*model name.*:", "", line, 1)
        print(f"CPU:{cpu_info}")
        # Check for and print GPU info if CUDA is available
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.init()  # Initialize CUDA to get accurate GPU info
        else:
            print("CUDA is not available.")
        print("Device:", self.device)

    def init_model(self):
        print("-------------------------------------------------")
        print("Initializing model...")
        self.model = Model(cfg["train"], cfg["model"], cfg["data"])
        self.model.to(device=self.device).eval()
        self.model.freeze()
        self.joints_to_vertices = JointsToVertices(mano_dir=cfg["data"]["mano_models_dir"], device=self.device)

    def loop(self):
        n_views = 8
        img_size = cfg["data"]["image_size"]
        x = torch.randn((1, n_views, 3, img_size, img_size), device=self.device)
        bbox = torch.randn((1, n_views, 4), device=self.device)
        cam_params = {
                        "intrinsic": torch.Tensor(1, n_views, 4).to(self.device),
                        "extrinsic": torch.Tensor(1, n_views, 4, 4).to(self.device)
                    }
        # flops, params = profile(self.model, (x, bbox, cam_params))

        with torch.no_grad():
            n_warmup_runs = 100
            print("\nJust warming up...\n")
            for _ in tqdm(range(n_warmup_runs)):
                out = self.model(x, bbox, cam_params)
                out["vertices"] = self.joints_to_vertices(out["joints_cam"][0].cpu().numpy())

            n_measurement_runs = 1000
            print(f"\nMeasuring inference speed as an average of {n_measurement_runs} runs.\n")
            times = []
            for _ in tqdm(range(n_measurement_runs)):
                start_time = time()
                out = self.model(x, bbox, cam_params)
                out["vertices"] = self.joints_to_vertices(out["joints_cam"][0].cpu().numpy())
                end_time = time()
                times.append(end_time - start_time)

        # Calculate average and last FPS
        average_fps = n_measurement_runs / sum(times)
        last_fps = 1 / times[-1]

        print("-------------------------------------------------")
        print(f"Batch size: {x.shape[0]}")
        print(f"Camera views: {n_views}")
        # print(f"GFlops: {flops/1e9:.3f}")
        print(f"Average FPS: {average_fps:.3f}")
        # print(f"Last FPS: {last_fps:.3f}")
        print(f"Average Inference Time: {(sum(times) / n_measurement_runs):.3f} seconds")
        # print(f"Last Inference Time: {times[-1]:.3f} seconds")
        print("-------------------------------------------------")

if __name__ == '__main__':
    InferenceSpeedTest()