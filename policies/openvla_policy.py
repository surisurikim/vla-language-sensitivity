"""
openvla_policy.py

OpenVLA policy wrapper for SimplerEnv.
Implements the model interface expected by SimplerEnv's maniskill2_evaluator:
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

# Prompt template used by OpenVLA
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
            model_path: HuggingFace model ID or local path to OpenVLA checkpoint.
            policy_setup: "google_robot" or "widowx_bridge".
                          Determines unnorm_key for action un-normalization.
            action_scale: Scalar multiplier for output actions.
            device: CUDA device string.
        """
        self.policy_setup = policy_setup
        self.action_scale = action_scale
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # unnorm_key maps to the dataset statistics used during OpenVLA training
        if policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data"
            self.sticky_gripper_num_repeat = 15
        elif policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig"
            self.sticky_gripper_num_repeat = 1
        else:
            raise ValueError(f"Unknown policy_setup: {policy_setup}")

        print(f"[OpenVLA] Loading processor from {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        print(f"[OpenVLA] Loading model from {model_path} (BF16)")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            attn_implementation="eager",   # use "flash_attention_2" if flash_attn is installed
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print("[OpenVLA] Model loaded successfully.")

        # Sticky gripper state (mimics how the Google Robot eval was done in the original paper)
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None

    def reset(self, task_description: str) -> None:
        """Called at the start of each episode."""
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    @torch.inference_mode()
    def step(self, image: np.ndarray, task_description: str):
        """
        Given an RGB image and task instruction, return the next robot action.

        Args:
            image: np.ndarray of shape (H, W, 3), dtype uint8
            task_description: language instruction string

        Returns:
            raw_action: np.ndarray of shape (7,) — raw output from OpenVLA
            action: dict with keys expected by SimplerEnv maniskill2_evaluator
                - "world_vector": np.ndarray (3,)
                - "rot_axangle": np.ndarray (3,)
                - "gripper": np.ndarray (1,)
                - "terminate_episode": np.ndarray (1,)
        """
        prompt = get_openvla_prompt(task_description)
        pil_image = Image.fromarray(image)

        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)
        raw_action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=False,
        )
        raw_action = raw_action.squeeze(0)  # (7,)

        # --- Parse raw action ---
        # OpenVLA action space: [dx, dy, dz, drx, dry, drz, gripper]
        # gripper: -1 = open, +1 = close  (or vice versa depending on setup)
        raw_action = {
            "world_vector": raw_action[:3],
            "rotation_delta": raw_action[3:6],
            "gripper": raw_action[6:7],
        }

        # Scale actions
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale

        # Convert rotation delta (roll-pitch-yaw) to axis-angle
        roll, pitch, yaw = raw_action["rotation_delta"]
        action["rot_axangle"] = np.array(euler2axangle(roll, pitch, yaw)) * self.action_scale

        # --- Sticky gripper logic (important for Google Robot eval) ---
        # The Google Robot needs the gripper command to be held for several steps
        if self.policy_setup == "google_robot":
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0.0])
            else:
                relative_gripper_action = (
                    raw_action["gripper"] - self.previous_gripper_action
                )
            self.previous_gripper_action = raw_action["gripper"]

            if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.gripper_action_repeat = 0

            if self.sticky_action_is_on:
                action["gripper"] = self.sticky_gripper_action
                self.gripper_action_repeat += 1
                if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                    self.sticky_action_is_on = False
                    self.gripper_action_repeat = 0
            else:
                action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["gripper"] > 0.0) - 1.0

        action["terminate_episode"] = np.array([0.0])

        # Convert to numpy float32 for SimplerEnv compatibility
        for k in action:
            if isinstance(action[k], torch.Tensor):
                action[k] = action[k].cpu().numpy().astype(np.float64)
            else:
                action[k] = np.array(action[k], dtype=np.float64)

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
        """Save a visualization of predicted actions over an episode."""
        predicted_actions = np.array(predicted_actions)  # (T, 7)
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
        for i, (ax, label) in enumerate(zip(axes.flat, labels)):
            ax.plot(predicted_actions[:, i])
            ax.set_title(label)
            ax.set_xlabel("timestep")
        axes.flat[-1].axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
