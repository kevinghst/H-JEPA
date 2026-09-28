"""HJEPA Model Implementation"""

from collections.abc import Sequence
import torch
from torch import nn


class HJEPA(nn.Module):
	"""Hierarchical container for multiple JEPA models.

	Args:
		jepas: Ordered JEPA modules where jepas[0] is level 1,
			jepas[1] is level 2, etc.
	"""

	def __init__(self, jepas: Sequence[nn.Module]):
		super().__init__()

		if not isinstance(jepas, Sequence):
			raise TypeError("jepas must be a sequence of nn.Module instances")
		if len(jepas) == 0:
			raise ValueError("jepas must contain at least one JEPA module")

		non_modules = [i for i, module in enumerate(jepas) if not isinstance(module, nn.Module)]
		if non_modules:
			raise TypeError(
				"All entries in jepas must be nn.Module instances; "
				f"invalid indices: {non_modules}"
			)

		# ModuleList ensures every JEPA is registered as a submodule.
		self.jepas = nn.ModuleList(jepas)

	@property
	def num_levels(self) -> int:
		return len(self.jepas)

	def get_level(self, level: int) -> nn.Module:
		"""Get a JEPA by 1-based level index (level=1 -> jepas[0])."""
		if level < 1 or level > self.num_levels:
			raise IndexError(
				f"level must be in [1, {self.num_levels}], got {level}"
			)
		return self.jepas[level - 1]

	def __getitem__(self, idx: int) -> nn.Module:
		return self.jepas[idx]

	def __len__(self) -> int:
		return self.num_levels

	def _level_uses_proprio(self, level: int) -> bool:
		encoder = getattr(self.jepas[level - 1], "encoder", None)
		return hasattr(encoder, "pixel_encoder") and hasattr(encoder, "proprio_encoder")

	def _get_level_input_key(self, level: int) -> str:
		if self._level_uses_proprio(level):
			return f"pixel_embed_{level - 1}"
		return f"embed_{level - 1}"

	def _get_level_proprio_key(self, level: int) -> str:
		return f"proprio_embed_{level - 1}"

	def _collect_level_info(self, info: dict, level: int) -> dict:
		suffix = f"_level{level}"
		level_info = {}
		for info_key, value in info.items():
			if info_key.endswith(suffix):
				level_info[info_key[: -len(suffix)]] = value
		return level_info

	def _add_level_target_aliases(self, info: dict, level: int) -> None:
		"""Keep suffixed target streams alongside unsuffixed encoder inputs."""
		for info_key, value in list(info.items()):
			if info_key.startswith("action"):
				continue
			if not torch.is_tensor(value) or value.ndim < 3:
				continue
			info[f"{info_key}_level{level}"] = value

	def _random_crop_level_outputs(
		self,
		output: dict,
		level: int,
		target_len: int,
		extra_keys: set[str] | None = None,
		start: int | None = None,
	) -> None:
		target_len = int(target_len)
		if target_len <= 0:
			return

		level_keys = {
			f"embed_{level}",
			f"pixel_embed_{level}",
			f"proprio_embed_{level}",
			f"action_{level}",
		}
		ref_keys = [f"embed_{level}", f"action_{level}"]
		if extra_keys:
			level_keys.update(extra_keys)
			ref_keys.extend(sorted(extra_keys))

		ref = None
		for ref_key in ref_keys:
			value = output.get(ref_key)
			if torch.is_tensor(value) and value.ndim >= 3:
				ref = value
				break

		if ref is None or ref.size(1) <= target_len:
			return

		seq_len = ref.size(1)
		if start is None:
			start = self._draw_crop_start(seq_len, target_len, ref.device)
		indices = torch.arange(start, start + target_len, device=ref.device)
		suffix = f"_level{level}"

		for output_key, value in list(output.items()):
			if not torch.is_tensor(value) or value.ndim < 3:
				continue
			if value.size(1) != seq_len:
				continue
			if output_key in level_keys or output_key.endswith(suffix):
				output[output_key] = value.index_select(1, indices)

	def _draw_crop_start(self, seq_len: int, target_len: int, device) -> int:
		return torch.randint(0, seq_len - target_len + 1, (), device=device).item()

	def _level_geometry(self, level: int) -> tuple[int, int]:
		jepa = self.jepas[level - 1]
		return int(getattr(jepa, "temporal_stride", 1)), int(getattr(jepa, "temporal_window_size", 1))

	def _sparse_level1_plan(self, seq_len: int, max_level: int, device):
		"""Draw every level's crop start up front and return the level-1 frames
		that any kept step at any level reads (directly or through the windows
		of the levels in between). Frames outside this set would be cropped away
		unseen, so the level-1 encoder can skip them."""
		lengths = {1: int(seq_len)}
		for level in range(2, max_level + 1):
			stride, window = self._level_geometry(level)
			lengths[level] = (lengths[level - 1] - window) // stride + 1

		crop_starts = {}
		kept = {}
		for level in range(1, max_level + 1):
			target_len = getattr(self.jepas[level - 1], "target_length", None)
			if target_len is None or int(target_len) <= 0 or lengths[level] <= int(target_len):
				crop_starts[level] = None
				kept[level] = set(range(lengths[level]))
				continue
			start = self._draw_crop_start(lengths[level], int(target_len), device)
			crop_starts[level] = start
			kept[level] = set(range(start, start + int(target_len)))

		needed = set(kept[max_level])
		for level in range(max_level, 1, -1):
			stride, window = self._level_geometry(level)
			needed = kept[level - 1] | {t * stride + k for t in needed for k in range(window)}
		return crop_starts, sorted(needed)

	def _encode_level1_sparse(self, info: dict, key: str, frame_indices) -> None:
		"""Run the level-1 encoder on a subset of frames and scatter the results
		into zero-filled full-length embed_1 / pixel_embed_1 / proprio_embed_1.
		Actions are encoded on the full stream (cheap, and higher-level pooling
		needs every chunk)."""
		ref = info[key]
		idx = torch.as_tensor(frame_indices, device=ref.device, dtype=torch.long)
		sparse_info = dict(info)
		sparse_info[key] = ref.index_select(1, idx)
		if "proprio" in info:
			sparse_info["proprio"] = info["proprio"].index_select(1, idx)

		level_out = self.jepas[0].encode(sparse_info, key=key)

		seq_len = ref.size(1)
		for output_name in ("embed", "pixel_embed", "proprio_embed"):
			sparse = level_out.get(f"{output_name}_0")
			if sparse is None:
				continue
			full = sparse.new_zeros(sparse.size(0), seq_len, *sparse.shape[2:])
			full[:, idx] = sparse
			info[f"{output_name}_1"] = full
		if "action_0" in level_out:
			info["action_1"] = level_out["action_0"]

	def _store_chunked_state_outputs(
		self,
		info: dict,
		level_out: dict,
		level: int,
	) -> None:
		ref = level_out.get("embed_0")
		if not torch.is_tensor(ref) or ref.ndim < 3:
			return

		current_len = ref.size(1)
		previous_level_suffix = f"_level{level - 1}"
		for info_key, value in level_out.items():
			if (
				info_key.startswith("action")
				or info_key.startswith("embed_")
				or info_key.startswith("pixel_embed_")
				or info_key.startswith("proprio_embed_")
				or info_key.startswith("proprio_input")
			):
				continue
			if not torch.is_tensor(value) or value.ndim < 3:
				continue

			if not info_key.endswith(previous_level_suffix):
				continue
			base_key = info_key[: -len(previous_level_suffix)]
			output_key = f"{base_key}_level{level}"

			if value.size(1) != current_len:
				raise ValueError(
					f"Cannot store '{output_key}': source '{info_key}' has "
					f"time length {value.size(1)}, but level {level} embeddings "
					f"have time length {current_len}."
				)
			info[output_key] = value

	def encode_hierarchical(
		self,
		info: dict,
		key: str = "pixels",
		levels_to_encode: int = -1,
		chunk_temporal_inputs: bool = False,
		start_level: int = 1,
	) -> dict:
     
		"""Encode inputs through all JEPA levels.

		Produces level-indexed keys: embed_1..embed_n and action_1..action_n.
		Level 1 uses the provided key (default: "pixels"); each later level uses
		the previous level's embed/action outputs as its input.

		Args:
			info: Input dictionary.
			key: Input key for level 1 encoding.
			levels_to_encode: Number of levels to encode.
				Use -1 to encode all levels.
		"""
  
		if not isinstance(info, dict):
			raise TypeError("info must be a dict")

		if key not in info:
			raise KeyError(f"Input key '{key}' not found in info")

		if levels_to_encode == -1:
			max_level = self.num_levels
		elif levels_to_encode < 1:
			raise ValueError("levels_to_encode must be -1 or >= 1")
		else:
			max_level = min(levels_to_encode, self.num_levels)

		original_action = info.get("action", None)
		original_proprio = info.get("proprio", None)
		had_original_proprio = "proprio" in info
		temporary_proprio_keys = set()

		for level in range(start_level, max_level + 1):
			jepa = self.jepas[level - 1]

			if level == 1:
				encode_key = key
				encode_info = info
			else:
				encode_key = self._get_level_input_key(level)
				action_key = f"action_{level - 1}"
				if encode_key not in info:
					raise KeyError(f"Missing '{encode_key}' for level {level} encoding")
				if action_key not in info:
					raise KeyError(f"Missing '{action_key}' for level {level} encoding")
				info["action"] = info[action_key]
				if self._level_uses_proprio(level):
					proprio_key = self._get_level_proprio_key(level)
					if proprio_key not in info:
						raise KeyError(
							f"Missing '{proprio_key}' for level {level} encoding"
						)
					model_proprio_key = getattr(jepa.encoder, "proprio_key", "proprio")
					info[model_proprio_key] = info[proprio_key]
					if model_proprio_key != "proprio":
						temporary_proprio_keys.add(model_proprio_key)
				encode_info = info

			level_out = jepa.encode(
				encode_info,
				key=encode_key,
				chunk_temporal_inputs=chunk_temporal_inputs and level > 1,
			)
			if chunk_temporal_inputs and level > 1:
				self._store_chunked_state_outputs(info, level_out, level)

			if "embed_0" not in level_out:
				raise KeyError(f"Level {level} JEPA encode did not return 'embed_0'")
			for output_name in ("embed", "pixel_embed", "proprio_embed"):
				output_key = f"{output_name}_0"
				if output_key in level_out:
					info[f"{output_name}_{level}"] = level_out[output_key]

			if "action_0" in level_out:
				info[f"action_{level}"] = level_out["action_0"]

		if original_action is not None:
			info["action"] = original_action
		if had_original_proprio:
			info["proprio"] = original_proprio
		else:
			info.pop("proprio", None)
		for temporary_key in temporary_proprio_keys:
			info.pop(temporary_key, None)

		return info

	def encode_hierarchical_per_level_inputs(
		self,
		info: dict,
		key: str = "pixels",
		levels_to_encode: int = -1,
		sparse_level1_encode: bool = True,
	) -> dict:
		"""Encode the level-1 sequence (f"{key}_level1", "action_level1") through
		all levels and randomly crop each level's outputs to its target length."""
		if not isinstance(info, dict):
			raise TypeError("info must be a dict")

		if levels_to_encode == -1:
			max_level = self.num_levels
		elif levels_to_encode < 1:
			raise ValueError("levels_to_encode must be -1 or >= 1")
		else:
			max_level = min(levels_to_encode, self.num_levels)

		output = dict(info)

		level_info = self._collect_level_info(info, 1)
		self._add_level_target_aliases(level_info, 1)
		level1_working_keys = set(level_info)
		if key not in level_info:
			raise KeyError(f"Missing '{key}_level1' for hierarchical encoding")
		if "action" not in level_info:
			raise KeyError("Missing 'action_level1' for hierarchical encoding")

		crop_starts = {}
		start_level = 1
		if sparse_level1_encode:
			crop_starts, frame_indices = self._sparse_level1_plan(
				level_info[key].size(1), max_level, level_info[key].device
			)
			self._encode_level1_sparse(level_info, key, frame_indices)
			start_level = 2

		level_out = self.encode_hierarchical(
			level_info,
			key=key,
			levels_to_encode=max_level,
			chunk_temporal_inputs=True,
			start_level=start_level,
		)
		internal_keys = level1_working_keys | {"embed_0", "action_0"}
		output.update(
			{
				output_key: value
				for output_key, value in level_out.items()
				if output_key not in internal_keys
			}
		)

		for level in range(1, max_level + 1):
			target_len = getattr(self.jepas[level - 1], "target_length", None)
			if target_len is None:
				continue
			extra_keys = level1_working_keys if level == 1 else None
			self._random_crop_level_outputs(
				output,
				level=level,
				target_len=target_len,
				extra_keys=extra_keys,
				start=crop_starts.get(level),
			)

		return output
