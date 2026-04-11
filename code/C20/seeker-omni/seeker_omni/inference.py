from pathlib import Path
from typing import Literal

import torch
from tokenizers import Tokenizer
from transformers import AutoImageProcessor, SiglipVisionModel
from torch.nn import functional as F
from seeker_omni.config import ExperimentConfig, load_yaml
from seeker_omni.model import SeekerOmniLM
from seeker_omni.model.resampler import PerceiverResampler
from seeker_omni.paths import TOKENIZER_DIR
from seeker_omni.steps.e2e.vision import load_rgb
from seeker_omni.train.checkpoint import load_checkpoint, latest_checkpoint


class SeekerOmniInference:
    def __init__(
            self,
            model_config: str | Path,
            checkpoint_path: str | Path | None,
            text_tokenizer: str | Path = TOKENIZER_DIR,
            vision_model: str = "google/siglip2-base-patch16-224",
            image_embed: Literal["avgpool", "resample"] = "avgpool",
            device: Literal["auto", "cuda", "cpu"] = "auto"
    ):
        """
        初始化推理器

        Args:
            model_config: 模型配置文件路径
            checkpoint_path: 模型检查点路径
            text_tokenizer: 文本分词器路径
            vision_model: 视觉模型名称或路径
            device: 运行设备 (auto, cuda, cpu)
        """
        # 加载配置
        self.cfg = ExperimentConfig.load(model_config)

        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 加载分词器
        self.tokenizer = Tokenizer.from_file(str(Path(text_tokenizer) / "tokenizer.json"))
        if self.tokenizer.get_vocab_size() != int(self.cfg.model.vocab_size):
            raise ValueError(f"text vocab mismatch: tokenizer={self.tokenizer.get_vocab_size()} cfg={self.cfg.model.vocab_size}")

        # 加载配置文件
        cfg_inf = load_yaml("configs/inference.yaml")

        # 加载视觉模型和处理器
        # 图像处理器为视觉模型准备输入特征，包括诸如调整大小、标准化以及转换为 PyTorch
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model)
        self.vision_model = SiglipVisionModel.from_pretrained(vision_model).to(self.device)
        self.vision_model.load_state_dict(torch.load(cfg_inf.get("vision_ckpt") , map_location=self.device))
        self.vision_model.eval()

        # 检查视觉模型配置
        patch_size = getattr(self.vision_model.config, "patch_size", None)
        if patch_size is None:
            raise ValueError("could not infer vision patch_size from vision.config")
        self.patch_size = int(patch_size)

        feat_dim = int(getattr(self.vision_model.config, "hidden_size", 0))
        if feat_dim != int(self.cfg.model.image_feat_dim):
            raise ValueError(
                f"vision feat_dim mismatch: vision={feat_dim} cfg.model.image_feat_dim={self.cfg.model.image_feat_dim}")

        # 根据 image_embed 初始化图片嵌入（用于处理图像特征）
        self.image_embed = image_embed
        if image_embed != "avgpool":
            self.resampler = PerceiverResampler(
                dim=feat_dim,
                num_latents=int(self.cfg.model.image_tokens),
                num_layers=2,
                num_heads=8,
                ff_mult=4,
            ).to(self.device)
            self.resampler.load_state_dict(torch.load(cfg_inf.get("resampler_ckpt"), map_location=self.device))
            self.resampler.eval()



        # 加载语言模型
        self.model = SeekerOmniLM(self.cfg.model).to(self.device)
        if checkpoint_path is None:
            checkpoint_path = latest_checkpoint(cfg_inf.get("out_dir"))
        load_checkpoint(checkpoint_path, model=self.model, optimizer=None)
        self.model.eval()

        # 获取特殊token
        self.pad_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.bos_id = self.tokenizer.token_to_id("<|im_start|>")
        self.eos_id = self.tokenizer.token_to_id("<|im_end|>")
        self.img_bos_id = self.tokenizer.token_to_id("<img_bos>")
        self.img_id = self.tokenizer.token_to_id("<img>")
        self.img_eos_id = self.tokenizer.token_to_id("<img_eos>")

        # 验证特殊token
        required_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<img_bos>", "<img>", "<img_eos>"]
        for token in required_tokens:
            if self.tokenizer.token_to_id(token) is None:
                raise ValueError(f"tokenizer missing required token: {token}")

    def _encode_text(self, text: str) -> list[int]:
        """编码文本为token ids"""
        return self.tokenizer.encode(text).ids

    def _process_image(self, image) -> torch.Tensor:
        """处理图像并提取特征"""
        px = self.vision_processor(images=image, return_tensors="pt").get("pixel_values")
        if px is None:
            raise RuntimeError("vision processor did not return pixel_values")
        px = px.to(self.device)

        # 提取特征
        with torch.no_grad():
            vout = self.vision_model(pixel_values=px)
            hs = vout.last_hidden_state

        # 移除 CLS token
        h = int(px.shape[-2])
        w = int(px.shape[-1])
        patch_h = max(1, h // int(self.patch_size))
        patch_w = max(1, w // int(self.patch_size))
        patch_count = int(patch_h * patch_w)
        if int(hs.shape[1]) == patch_count + 1:
            hs = hs[:, 1:, :]

        # 根据 image_embed 进行嵌入
        target_tokens = int(self.cfg.model.image_tokens)
        if self.image_embed == "avgpool":
            image_feats = F.adaptive_avg_pool1d(hs.transpose(1, 2), target_tokens).transpose(1, 2)
        else:
            image_feats = self.resampler(hs)

        return image_feats

    def _build_input(self, prompt: str, image=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        构建模型输入

        Args:
            prompt: 用户提示文本
            image: 图像输入（可选）

        Returns:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            image_feats: 图像特征（如果提供了图像）
        """
        # 构建输入序列
        system = "你是一个只用中文回答的助手。"
        nl = self._encode_text("\n")

        # 拼接系统提示
        tokens = [self.bos_id] + self._encode_text("system\n") + self._encode_text(system) + [self.eos_id] + nl
        # 拼接用户提示
        tokens += [self.bos_id] + self._encode_text("user\n") + self._encode_text(prompt) + nl
        # 图像占位
        tokens += [self.img_bos_id] + [self.img_id] * int(self.cfg.model.image_tokens) + [self.img_eos_id]
        tokens += [self.eos_id] + nl
        # 助手回复前缀
        tokens += [self.bos_id] + self._encode_text("assistant\n")

        # 转换为张量
        inputs_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # 构建注意力掩码
        attention_mask = torch.ones_like(inputs_ids, dtype=torch.float32, device=self.device)

        # 处理图像
        image_feats = None
        if image is not None:
            image_feats = self._process_image(image)

        return inputs_ids, attention_mask, image_feats

    def generate(
            self,
            prompt: str,
            image=None,
            max_new_tokens: int = 64,
            temperature: float = 0.7,
            top_p: float = 0.95,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            repetition_window: int = 256,
            no_repeat_ngram_size: int = 0,
    ) -> str:
        """
        生成文本响应

        Args:
            prompt: 用户提示文本
            image: 图像输入（可选）
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_p: 核采样参数
            top_k: _top-k采样参数
            repetition_penalty: 重复惩罚
            repetition_window: 重复惩罚窗口大小
            no_repeat_ngram_size: n-gram 循环禁止
        Returns:
            生成的文本响应
        """
        with torch.no_grad():
            # 构建输入
            input_ids, attention_mask, image_feats = self._build_input(prompt, image)

            # 生成文本
            output_ids = self.model.generate_text(
                input_ids,
                attention_mask=attention_mask,
                image_feats=image_feats,
                max_new_tokens=max_new_tokens,
                eos_id=self.eos_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                no_repeat_ngram_size=no_repeat_ngram_size
            )

            # 解码输出
            # 找到assistant部分的开始
            assistant_start = len(input_ids[0])

            # 找到eos的位置
            eos_positions = [i for i, token_id in enumerate(output_ids[0][assistant_start:]) if token_id == self.eos_id]
            if eos_positions:
                output_ids = output_ids[0][assistant_start:assistant_start + eos_positions[0]]
            else:
                output_ids = output_ids[0][assistant_start:]

            # 解码文本
            response = self.tokenizer.decode(output_ids.tolist())
            return response.strip()

def main():
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description="Seeker Omni 推理")
    parser.add_argument("--config", type=str, default="configs/stages/sft_text.yaml")
    parser.add_argument("--checkpoint", type=str, required=False, help="模型检查点路径")
    parser.add_argument("--prompt", type=str, default="请描述图片内容", help="提示文本")
    parser.add_argument("--image", type=str, help="图像路径")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--device", type=str, default="auto", help="运行设备")

    args = parser.parse_args()

    inference = SeekerOmniInference(
        model_config=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 加载图像
    image = None
    if args.image:
        image = [load_rgb(args.image)]

    response = inference.generate(
        prompt=args.prompt,
        image=image,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("输入提示:", args.prompt)
    if args.image:
        print("图像路径:", args.image)
    print("\n生成响应:")
    print(response)

if __name__ == "__main__":
    main()