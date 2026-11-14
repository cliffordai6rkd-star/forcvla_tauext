import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_forcevla_example() -> dict:
    """Creates a random example with separate proprioception and joint-torque inputs."""
    return {
        "state": np.ones((7,)),  # seven robot proprioception channels
        "tau_ext": np.ones((7,)),  # external joint torques
        "image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8), 
        "wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8), 
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Images must be single RGB frames in HWC or CHW format, got {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError(f"Images must have three RGB channels and nonempty dimensions, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Float RGB images must be finite and in [0, 1]; use uint8 for [0, 255]")
        image = (255 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"Images must be uint8 or float RGB in [0, 1], got {image.dtype}")
    return image


@dataclasses.dataclass(frozen=True)
class ForcevlaServingInputs(transforms.DataTransformFn):
    """Validate local v3 deployment inputs before normalization.

    The dataset already contains compensated, filtered and gated tau_ext.
    Reapplying the contact gate online is idempotent; dynamics compensation
    and causal filtering must still be performed by the robot client.
    """

    image_keys: tuple[str, ...]
    tau_ext_contact_threshold: float | None = None

    def __call__(self, data: dict) -> dict:
        for key in self.image_keys:
            if key not in data:
                raise ValueError(
                    f"Missing required camera {key!r}; send front as 'image' and right wrist as 'right_wrist_image'"
                )
        if "wrist_image" in data and "wrist_image" not in self.image_keys:
            raise ValueError(
                "This checkpoint has no left camera: omit 'wrist_image'; send right wrist as 'right_wrist_image'"
            )
        result = dict(data)
        for key in ("state", "tau_ext"):
            values = np.asarray(data[key])
            if values.shape != (7,) or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f"{key} must be a single real seven-vector, without a batch dimension")
            if not np.isfinite(values).all():
                raise ValueError(f"{key} contains NaN or Inf")
            result[key] = values
        if (
            self.tau_ext_contact_threshold is not None
            and np.abs(result["tau_ext"].astype(np.float64)).sum() < self.tau_ext_contact_threshold
        ):
            result["tau_ext"] = np.zeros_like(result["tau_ext"])
        return result


@dataclasses.dataclass(frozen=True)
class Forcevla_inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Route external joint torques separately from the robot's proprioception.
        # Discard any legacy wrench channels appended to observation.state.
        proprio = np.asarray(data["state"])
        tau_ext = np.asarray(data["tau_ext"])
        if proprio.ndim < 1 or proprio.shape[-1] < 7:
            raise ValueError("state must contain robot proprioception in its first 7 channels")
        if tau_ext.ndim < 1 or tau_ext.shape[-1] != 7:
            raise ValueError(f"tau_ext must have 7 channels, got shape {tau_ext.shape}")
        if proprio.shape[:-1] != tau_ext.shape[:-1]:
            raise ValueError("state and tau_ext must have matching batch dimensions")
        if not np.isfinite(proprio).all() or not np.isfinite(tau_ext).all():
            raise ValueError("state and tau_ext must contain only finite values")
        if self.action_dim < 14:
            raise ValueError("ForceVLA requires action_dim >= 14 to hold proprioception and tau_ext")
        state = np.concatenate([proprio[..., :7], tau_ext], axis=-1)
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # a missing wrist view below.
        base_image = _parse_image(data["image"])
        left_wrist_image = _parse_image(data["wrist_image"]) if "wrist_image" in data else np.zeros_like(base_image)
        right_wrist_image = (
            _parse_image(data["right_wrist_image"]) if "right_wrist_image" in data else np.zeros_like(base_image)
        )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.bool_("wrist_image" in data or not mask_padding),
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.bool_("right_wrist_image" in data or not mask_padding),
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs
    
@dataclasses.dataclass(frozen=True)
class Forcevla_outputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # The first 7 channels are robot actions; local v3 joint configs use all 7 as joints.
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[1] < 7 or not np.isfinite(actions).all():
            raise ValueError("ForceVLA returned invalid or nonfinite joint actions")
        return {"actions": actions[:, :7]}
