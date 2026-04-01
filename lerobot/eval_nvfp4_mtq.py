import torch
import argparse
import mtq
from mtq.configs import NVFP4_DEFAULT_CFG
from lerobot.common.policies.factory import make_policy
from lerobot.common.envs.factory import make_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_paths", nargs='+', required=True)
    parser.add_argument("--env_name", type=str, default="lerobot/pusht_image")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    
    for path in args.policy_paths:
        print(f"🚀 Loading model for MTQ NVFP4: {path}")
        policy = make_policy(path, device=device)
        policy.eval()
        
        # 🎯 mtq 라이브러리를 사용하여 NVFP4 적용
        # mtq.quantize는 모델의 Linear 레이어들을 NVFP4 설정으로 자동 변환합니다.
        quantized_policy = mtq.quantize(policy, config=NVFP4_DEFAULT_CFG)
        
        env = make_env(args.env_name, n_envs=1)
        obs, info = env.reset()
        
        print(f"🔥 NVFP4 Quantization Applied using MTQ for {path}")
        # (Evaluation Loop 실행 코드)

if __name__ == "__main__":
    main()