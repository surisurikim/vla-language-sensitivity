"""
openvla_policy.py

SimplerEnv에서 OpenVLA를 사용하기 위한 policy wrapper.
SimplerEnv의 maniskill2_evaluator가 요구하는 모델 인터페이스를 구현한다:
  - model.reset(task_description)
  - model.step(image, task_description) -> (raw_action, action_dict)
  - model.visualize_epoch(predicted_actions, images, save_path)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor


OPENVLA_MODEL_ID = "openvla/openvla-7b"

# OpenVLA에서 사용하는 프롬프트 형식
def get_openvla_prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


class OpenVLAInference:
    def __init__(
        self,
        model_path: str = OPENVLA_MODEL_ID,
        policy_setup: str = "google_robot",
        action_scale: float = 1.0,
        device: str = "cuda:0",
    ):
        """
        Args:
            model_path: HuggingFace 모델 ID 또는 로컬 체크포인트 경로
            policy_setup: "google_robot" 또는 "widowx_bridge"
                          action 역정규화에 사용할 unnorm_key를 결정함
            action_scale: action 출력값에 곱할 스케일 인수
            device: 사용할 CUDA 디바이스 문자열
        """
        self.policy_setup = policy_setup
        self.action_scale = action_scale
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # unnorm_key: OpenVLA 학습 시 사용된 데이터셋 통계와 매핑됨
        # action을 실제 로봇 단위로 역정규화할 때 사용
        if policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data"
            self.sticky_gripper_num_repeat = 15  # 그리퍼 명령을 유지할 스텝 수
        elif policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig"
            self.sticky_gripper_num_repeat = 1
        else:
            raise ValueError(f"알 수 없는 policy_setup: {policy_setup}")

        print(f"[OpenVLA] 프로세서 로딩 중: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"[OpenVLA] 모델 로딩 중: {model_path} (BF16)")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            # transformers 4.40.1에서 "eager"는 attention mask 크기 불일치 버그가 있음
            # PyTorch 2.0+의 기본값인 SDPA(Scaled Dot Product Attention)를 사용
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print("[OpenVLA] 모델 로딩 완료.")

        # sticky gripper 상태 변수
        # Google Robot 평가에서는 그리퍼 명령을 여러 스텝 동안 유지해야 함 (논문 설정)
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None

    def reset(self, task_description: str) -> None:
        """에피소드 시작 시 호출. 내부 상태를 초기화한다."""
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    @torch.inference_mode()
    def step(self, image: np.ndarray, task_description: str):
        """
        RGB 이미지와 언어 명령을 받아 다음 로봇 action을 반환한다.

        Args:
            image: shape (H, W, 3), dtype uint8의 RGB 이미지
            task_description: 언어 명령 문자열

        Returns:
            raw_action: shape (7,)의 OpenVLA 원시 출력값
            action: SimplerEnv maniskill2_evaluator가 요구하는 형식의 딕셔너리
                - "world_vector": np.ndarray (3,) — 위치 델타 (x, y, z)
                - "rot_axangle": np.ndarray (3,) — 회전 델타 (axis-angle)
                - "gripper": np.ndarray (1,) — 그리퍼 명령
                - "terminate_episode": np.ndarray (1,) — 에피소드 종료 신호
        """
        prompt = get_openvla_prompt(task_description)
        pil_image = Image.fromarray(image)

        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)
        raw_action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=False,
        )
        raw_action = raw_action.squeeze(0)  # 배치 차원 제거 → shape (7,)

        # OpenVLA action 공간: [dx, dy, dz, drx, dry, drz, gripper]
        # gripper: -1 = 열림, +1 = 닫힘 (setup에 따라 다를 수 있음)
        raw_action = {
            "world_vector": raw_action[:3],
            "rotation_delta": raw_action[3:6],
            "gripper": raw_action[6:7],
        }

        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale

        # 회전 델타: roll-pitch-yaw → axis-angle 변환 (SimplerEnv 요구 형식)
        roll, pitch, yaw = raw_action["rotation_delta"]
        action["rot_axangle"] = np.array(euler2axangle(roll, pitch, yaw)) * self.action_scale

        # --- Sticky gripper 처리 ---
        # Google Robot은 그리퍼 변화 명령을 여러 스텝 유지해야 실제 동작이 됨
        if self.policy_setup == "google_robot":
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0.0])
            else:
                relative_gripper_action = (
                    raw_action["gripper"] - self.previous_gripper_action
                )
            self.previous_gripper_action = raw_action["gripper"]

            # 그리퍼 변화량이 임계값을 넘으면 sticky 모드 시작
            if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.gripper_action_repeat = 0

            if self.sticky_action_is_on:
                action["gripper"] = self.sticky_gripper_action
                self.gripper_action_repeat += 1
                # 지정된 스텝 수만큼 반복 후 sticky 모드 종료
                if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                    self.sticky_action_is_on = False
                    self.gripper_action_repeat = 0
            else:
                action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            # widowx는 이진 그리퍼 제어: 0보다 크면 닫힘(+1), 아니면 열림(-1)
            action["gripper"] = 2.0 * (raw_action["gripper"] > 0.0) - 1.0

        action["terminate_episode"] = np.array([0.0])

        # SimplerEnv 호환을 위해 모든 값을 numpy float64로 변환
        for k in action:
            if isinstance(action[k], torch.Tensor):
                action[k] = action[k].cpu().numpy().astype(np.float64)
            else:
                action[k] = np.array(action[k], dtype=np.float64)

        # 로깅/시각화를 위한 원시 action numpy 배열 (shape: 7,)
        raw_action_np = np.concatenate([
            raw_action["world_vector"] if isinstance(raw_action["world_vector"], np.ndarray)
            else raw_action["world_vector"].cpu().numpy(),
            raw_action["rotation_delta"] if isinstance(raw_action["rotation_delta"], np.ndarray)
            else raw_action["rotation_delta"].cpu().numpy(),
            raw_action["gripper"] if isinstance(raw_action["gripper"], np.ndarray)
            else raw_action["gripper"].cpu().numpy(),
        ])

        return raw_action_np, action

    def visualize_epoch(self, predicted_actions, images, save_path: str) -> None:
        """한 에피소드 동안의 predicted action을 시각화하여 저장한다."""
        predicted_actions = np.array(predicted_actions)  # shape: (타임스텝 수, 7)
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        for i, (ax, label) in enumerate(zip(axes.flat, labels)):
            ax.plot(predicted_actions[:, i])
            ax.set_title(label)
            ax.set_xlabel("타임스텝")
        axes.flat[-1].axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
