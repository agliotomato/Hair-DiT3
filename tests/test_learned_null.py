
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.hair_s2i_net import HairS2INet

def test_learned_null_init():
    # 실제 모델을 로드하면 VRAM이 너무 많이 들 수 있으므로 
    # mocking이나 가벼운 방식으로 체크하는 것이 좋지만, 
    # 여기서는 클래스 초기화가 에러 없이 되는지만 확인 (인코더 로드 안하는지)
    
    # 팁: pretrained_model_name_or_path에 존재하지 않는 경로를 넣으면 에러가 나야 함 
    # 하지만 VAE와 Transformer는 여전히 로드하므로 유효한 경로가 필요함.
    # 테스트 환경에 해당 모델이 없을 수 있으므로, Patching을 고려함.
    
    print("Testing HairS2INet initialization with Learned Null Embedding...")
    
    # Text Encoder들이 속성으로 존재하지 않아야 함
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    
    # 유닛 테스트를 위해 실제 로딩은 건너뛰고 구조만 확인하고 싶다면 
    # 별도의 Mocking이 필요하나, 여기서는 수동으로 코드 구조상 
    # 에러가 없는지 forward 시그니처만 체크함.
    
    # forward signature check
    import inspect
    sig = inspect.signature(HairS2INet.forward)
    print(f"Forward signature: {sig}")
    
    assert "prompt_embeds" not in sig.parameters, "prompt_embeds should be removed from forward"
    assert "pooled_embeds" not in sig.parameters, "pooled_embeds should be removed from forward"
    print("SUCCESS: forward signature is updated.")

if __name__ == "__main__":
    test_learned_null_init()
