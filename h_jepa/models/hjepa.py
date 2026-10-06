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
		"""Number of levels in the hierarchy."""
		return len(self.jepas)

	def get_level(self, level: int) -> nn.Module:
		"""Get a JEPA by 1-based level index (level=1 -> jepas[0])."""
		if level < 1 or level > self.num_levels:
			raise IndexError(
				f"level must be in [1, {self.num_levels}], got {level}"
			)
		return self.jepas[level - 1]

	def __getitem__(self, idx: int) -> nn.Module:
		"""Get a JEPA by 0-based index (idx=0 -> level 1)."""
		return self.jepas[idx]

	def __len__(self) -> int:
		"""Number of levels in the hierarchy."""
		return self.num_levels

	def _level_uses_proprio(self, level: int) -> bool:
		"""Whether this level encodes pixel and proprio streams with a fusion encoder."""
		encoder = getattr(self.jepas[level - 1], "encoder", None)
		return hasattr(encoder, "pixel_encoder") and hasattr(encoder, "proprio_encoder")

	def _get_level_input_key(self, level: int) -> str:
		"""Key of the previous level's output this level encodes: pixel_embed_{level-1}
		for fusion (proprio) levels, embed_{level-1} otherwise."""
		if self._level_uses_proprio(level):
			return f"pixel_embed_{level - 1}"
		return f"embed_{level - 1}"

	def _collect_level_info(self, info: dict, level: int) -> dict:
		"""Return the entries of info with a "_level{level}" suffix, with the suffix
		stripped (e.g. pixels_level1 -> pixels)."""
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
		"""Crop this level's outputs in place to target_len time steps: embed/
		pixel_embed/proprio_embed/action_{level}, every "*_level{level}" key and
		extra_keys, all at the same offset (start, or a random one if None).
		Only tensors whose time length matches the level's are cropped."""
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
		"""Uniformly draw the start of a target_len crop of a seq_len sequence."""
		return torch.randint(0, seq_len - target_len + 1, (), device=device).item()

	def _level_geometry(self, level: int) -> tuple[int, int]:
		"""(stride, window_size) with which this level chunks the level below."""
		jepa = self.jepas[level - 1]
		return int(getattr(jepa, "temporal_stride", 1)), int(getattr(jepa, "temporal_window_size", 1))

	def _draw_crops_and_level1_frames(self, seq_len: int, max_level: int, device):
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

	def _encode_level1(self, info: dict, key: str, frame_indices) -> None:
		"""Run the level-1 encoder on the given frames (all of them when dense) and
		scatter the results into zero-filled full-length embed_1 / pixel_embed_1 /
		proprio_embed_1.
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
		"""Copy the target streams that JEPA.encode chunked to this level's time
		steps (level_out's "*_level{level-1}" keys) into info as "*_level{level}"."""
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

	def encode_hierarchical(self, info: dict, sparse_level1_encode: bool = True) -> dict:
		"""Encode the level-1 sequence ("pixels_level1", "action_level1") through
		all levels and randomly crop each level's outputs to its target length."""
		level_info = self._collect_level_info(info, 1)
		self._add_level_target_aliases(level_info, 1)
		level1_working_keys = set(level_info)

		seq_len = level_info["pixels"].size(1)
		if sparse_level1_encode:
			crop_starts, frame_indices = self._draw_crops_and_level1_frames(
				seq_len, self.num_levels, level_info["pixels"].device
			)
		else:
			crop_starts, frame_indices = {}, range(seq_len)
		self._encode_level1(level_info, "pixels", frame_indices)

		for level in range(2, self.num_levels + 1):
			jepa = self.jepas[level - 1]
			level_info["action"] = level_info[f"action_{level - 1}"]
			if self._level_uses_proprio(level):
				level_info[jepa.encoder.proprio_key] = level_info[f"proprio_embed_{level - 1}"]

			level_out = jepa.encode(
				level_info,
				key=self._get_level_input_key(level),
				chunk_temporal_inputs=True,
			)
			self._store_chunked_state_outputs(level_info, level_out, level)
			for output_name in ("embed", "pixel_embed", "proprio_embed"):
				if f"{output_name}_0" in level_out:
					level_info[f"{output_name}_{level}"] = level_out[f"{output_name}_0"]
			level_info[f"action_{level}"] = level_out["action_0"]

		output = dict(info)
		output.update(
			{
				output_key: value
				for output_key, value in level_info.items()
				if output_key not in level1_working_keys and output_key != "proprio_input"
			}
		)

		for level in range(1, self.num_levels + 1):
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
