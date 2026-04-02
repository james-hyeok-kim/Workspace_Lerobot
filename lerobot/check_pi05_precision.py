import torch
import json
import argparse
import os
from huggingface_hub import snapshot_download

def inspect_precision_from_weights(model_id, output_file):
    print(f"📦 [INFO] 로컬 가중치 파일 직접 분석 시작: {model_id}")
    
    try:
        # 1. 모델 파일을 로컬로 확보 (이미 있으면 바로 경로 반환)
        model_path = snapshot_download(
            model_id, 
            allow_patterns=["*.safetensors", "*.bin", "config.json"],
            local_files_only=False # 필요시 다운로드
        )
        
        # 2. 가중치 파일 찾기 (safetensors 우선, 그 다음 bin)
        weight_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.bin')]
        if not weight_files:
            print("❌ [ERROR] 가중치 파일을 찾을 수 없습니다.")
            return

        layer_data = []
        summary = {"total_parameters": 0, "dtype_counts": {}, "bit_distribution": {}}

        for wf in weight_files:
            file_path = os.path.join(model_path, wf)
            print(f"📖 [INFO] 파일 읽는 중: {wf}")
            
            # 3. 가중치 로드 (메모리 절약을 위해 cpu로 로드)
            if wf.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(file_path, device="cpu")
            else:
                state_dict = torch.load(file_path, map_files="cpu")

            # 4. 각 텐서(레이어)의 정보 추출
            for name, tensor in state_dict.items():
                dtype_str = str(tensor.dtype)
                bits = tensor.element_size() * 8
                numel = tensor.numel()
                
                # 카테고리 분류
                category = "Other"
                if "vision_tower" in name or "vit" in name: category = "Vision Encoder"
                elif "backbone" in name or "model.layers" in name: category = "Transformer Backbone"
                elif "action_head" in name: category = "Action Head"
                
                layer_data.append({
                    "name": name, "category": category, "dtype": dtype_str, 
                    "bits": bits, "numel": numel
                })

                summary["total_parameters"] += numel
                summary["dtype_counts"][dtype_str] = summary["dtype_counts"].get(dtype_str, 0) + 1
                summary["bit_distribution"][bits] = summary["bit_distribution"].get(bits, 0) + 1

        # 5. 결과 저장
        final_output = {"model_id": model_id, "summary": summary, "layers": layer_data}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4)
        
        print(f"✅ [SUCCESS] 분석 완료! 결과 파일: {output_file}")
        print(f"📊 요약: {summary['dtype_counts']}")

    except Exception as e:
        print(f"❌ [ERROR] 분석 중 치명적 오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    inspect_precision_from_weights(args.model_id, args.output)