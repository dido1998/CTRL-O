"""Implementations of perceptual grouping algorithms.

We denote methods that group input feature together into slots of objects
(either unconditionally) or via additional conditioning signals as perceptual
grouping modules.
"""
import math
from typing import Any, Callable, Dict, Optional, Union

import numpy
import torch
from sklearn import cluster
from torch import nn

import ocl.typing
from ocl.utils.tensor_ops import pixel_norm


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        slot_update: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        use_gru: bool = True,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_cosine_attention: bool = False,
        use_input_norm: bool = True,
        dual_conditioning: bool = False,
        point_conditioning: bool = False,
        lang_conditioning: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation
        self.use_cosine_attention = use_cosine_attention
        self.dual_conditioning = dual_conditioning
        self.point_conditioning = point_conditioning
        self.lang_conditioning = lang_conditioning

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.norm_slots = nn.LayerNorm(dim)
        self.dim = dim
        qdim = dim

        if self.dual_conditioning:
            self.add_dim = 2 + 128
            qdim += self.add_dim
        elif self.point_conditioning:
            self.add_dim = 2
            qdim += self.add_dim
        elif self.lang_conditioning:
            self.add_dim = 256
            qdim += self.add_dim

        self.qdim = qdim

        self.to_q = nn.Linear(qdim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        if use_gru:
            if slot_update is not None:
                raise ValueError("`use_gru` and `slot_update` are mutually exclusive")
            self.gru = nn.GRUCell(self.kvq_dim, dim)
        else:
            self.gru = None
            if isinstance(slot_update, nn.Module):
                self.to_update = slot_update
            elif callable(slot_update):
                self.to_update = slot_update()
            elif self.n_heads > 1 or self.kvq_dim != dim:
                self.to_update = nn.Linear(self.kvq_dim, dim, bias=use_projection_bias)
            else:
                self.to_update = nn.Identity()

        if use_input_norm:
            self.norm_input = nn.LayerNorm(feature_dim)
        else:
            self.norm_input = nn.Identity()

        if isinstance(ff_mlp, nn.Module):
            self.ff_mlp = ff_mlp
        elif callable(ff_mlp):
            self.ff_mlp = ff_mlp()
        elif ff_mlp is None:
            self.ff_mlp = nn.Identity()
        else:
            raise ValueError("ff_mlp needs to be nn.Module or callable returning nn.Module")

    def step(
        self,
        slots,
        k,
        v,
        masks=None,
        point_conditioning=None,
        dual_conditioning=None,
        lang_conditioning=None,
    ):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        if self.dual_conditioning:
            if dual_conditioning is None:
                slots = torch.cat(
                    [slots, torch.zeros(bs, n_slots, self.add_dim).to(slots.device)], dim=-1
                )
            else:
                slots = torch.cat([slots, dual_conditioning], dim=-1)
        elif self.point_conditioning:
            if point_conditioning is None:
                slots = torch.cat(
                    [slots, torch.zeros(bs, n_slots, self.add_dim).to(slots.device)], dim=-1
                )
            else:
                slots = torch.cat([slots, point_conditioning], dim=-1)
        elif self.lang_conditioning:
            if lang_conditioning is None:
                slots = torch.cat(
                    [slots, torch.zeros(bs, n_slots, self.add_dim).to(slots.device)], dim=-1
                )
            else:
                slots = torch.cat([slots, lang_conditioning], dim=-1)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        if self.use_cosine_attention:
            q = pixel_norm(q)  # Scale such that length is sqrt(D)
            k = pixel_norm(k)  # Scale such that length is sqrt(D)
        q = q * self.scale

        dots = torch.einsum("bihd,bjhd->bihj", q, k)
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn).flatten(-2, -1)

        if self.gru is not None:
            slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))
            slots = slots.reshape(updates.shape[0], -1, self.dim)
        elif isinstance(self.to_update, (nn.Identity, nn.Linear)):
            slots = slots_prev + self.to_update(updates)
        else:
            slots = self.to_update(updates, slots_prev)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(
        self,
        n_iters,
        slots,
        k,
        v,
        masks=None,
        point_conditioning=None,
        dual_conditioning=None,
        lang_conditioning=None,
    ):
        attn = None
        attn_list = []
        for _ in range(n_iters):
            slots, attn = self.step(
                slots, k, v, masks, point_conditioning, dual_conditioning, lang_conditioning
            )
            attn_list.append(attn)
        return slots, attn, attn_list

    def forward(
        self,
        inputs: torch.Tensor,
        conditioning: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        point_conditioning=None,
        dual_conditioning=None,
        lang_conditioning=None,
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, _, attn_list = self.iterate(
                self.iters - 1,
                slots,
                k,
                v,
                masks,
                point_conditioning,
                dual_conditioning,
                lang_conditioning,
            )
            slots, attn = self.step(
                slots.detach(), k, v, masks, point_conditioning, dual_conditioning, lang_conditioning
            )
            attn_list.append(attn)
        else:
            slots, attn, attn_list = self.iterate(
                self.iters,
                slots,
                k,
                v,
                masks,
                point_conditioning,
                dual_conditioning,
                lang_conditioning,
            )

        return slots, attn, attn_list


class SlotAttentionGrouping(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        n_blocks: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        slot_update: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_gru: bool = True,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_cosine_attention: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            feature_dim: Dimensionality of features to slot attention (after positional encoding).
            object_dim: Dimensionality of slots.
            kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
                `object_dim` is used.
            n_heads: Number of heads slot attention uses.
            n_blocks: Number of slot attention modules to chain.
            iters: Number of slot attention iterations.
            eps: Epsilon in slot attention.
            ff_mlp: Optional module applied slot-wise after GRU update.
            slot_update: Update module that applies attention updates to previous slots. Signature
                should be (update, previous_slots) -> updated_slots
            positional_embedding: Optional module applied to the features before slot attention,
                adding positional encoding.
            use_gru: Wether to use a GRU for updating slots, or a simple residual update.
            use_projection_bias: Whether to use biases in key, value, query projections.
            use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
                performs one more iteration of slot attention that is used for the gradient step
                after `iters` iterations of slot attention without gradients. Faster and more memory
                efficient than the standard version, but can not backpropagate gradients to the
                conditioning input.
            use_cosine_attention: Use cosine attention.
            use_empty_slot_for_masked_slots: Replace slots masked with a learnt empty slot vector.
        """
        super().__init__()
        self._object_dim = object_dim

        def make_slot_attention(use_input_norm=True):
            return SlotAttention(
                dim=object_dim,
                feature_dim=feature_dim,
                kvq_dim=kvq_dim,
                n_heads=n_heads,
                iters=iters,
                eps=eps,
                ff_mlp=ff_mlp,
                slot_update=slot_update,
                use_gru=use_gru,
                use_projection_bias=use_projection_bias,
                use_cosine_attention=use_cosine_attention,
                use_implicit_differentiation=use_implicit_differentiation,
                use_input_norm=use_input_norm,
            )

        if n_blocks == 1:
            self.slot_attention = make_slot_attention()
            self.norm_input = nn.Identity()
        else:
            self.slot_attention = nn.ModuleList(
                [make_slot_attention(use_input_norm=False) for _ in range(n_blocks)]
            )
            self.norm_input = nn.LayerNorm(feature_dim)

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.positional_embedding:
            features = self.positional_embedding(feature.features, feature.positions)
        else:
            features = feature.features

        features = self.norm_input(features)
        if isinstance(self.slot_attention, SlotAttention):
            slots, attn, _ = self.slot_attention(features, conditioning, slot_mask)
        else:
            slots = conditioning
            for slot_att in self.slot_attention:
                slots, attn, _ = slot_att(features, slots, slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=attn, is_empty=slot_mask
        )


class ControllableSlotAttentionGrouping(nn.Module):
    """Implementation of SlotAttention for perceptual grouping."""

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        n_blocks: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        slot_update: Optional[Union[nn.Module, Callable[[], nn.Module]]] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_gru: bool = True,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_cosine_attention: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        point_conditioning: bool = False,
        dual_conditioning: bool = False,
        lang_conditioning: bool = False,
    ):
        """Initialize Slot Attention Grouping.

        Args:
            feature_dim: Dimensionality of features to slot attention (after positional encoding).
            object_dim: Dimensionality of slots.
            kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
                `object_dim` is used.
            n_heads: Number of heads slot attention uses.
            n_blocks: Number of slot attention modules to chain.
            iters: Number of slot attention iterations.
            eps: Epsilon in slot attention.
            ff_mlp: Optional module applied slot-wise after GRU update.
            slot_update: Update module that applies attention updates to previous slots. Signature
                should be (update, previous_slots) -> updated_slots
            positional_embedding: Optional module applied to the features before slot attention,
                adding positional encoding.
            use_gru: Wether to use a GRU for updating slots, or a simple residual update.
            use_projection_bias: Whether to use biases in key, value, query projections.
            use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
                performs one more iteration of slot attention that is used for the gradient step
                after `iters` iterations of slot attention without gradients. Faster and more memory
                efficient than the standard version, but can not backpropagate gradients to the
                conditioning input.
            use_cosine_attention: Use cosine attention.
            use_empty_slot_for_masked_slots: Replace slots masked with a learnt empty slot vector.
        """
        super().__init__()
        self._object_dim = object_dim
        self.dual_conditioning = dual_conditioning
        self.point_conditioning = point_conditioning
        self.lang_conditioning = lang_conditioning

        def make_slot_attention(use_input_norm=True):
            return SlotAttention(
                dim=object_dim,
                feature_dim=feature_dim,
                kvq_dim=kvq_dim,
                n_heads=n_heads,
                iters=iters,
                eps=eps,
                ff_mlp=ff_mlp,
                slot_update=slot_update,
                use_gru=use_gru,
                use_projection_bias=use_projection_bias,
                use_cosine_attention=use_cosine_attention,
                use_implicit_differentiation=use_implicit_differentiation,
                use_input_norm=use_input_norm,
                dual_conditioning=dual_conditioning,
                point_conditioning=point_conditioning,
                lang_conditioning=lang_conditioning,
            )

        if n_blocks == 1:
            self.slot_attention = make_slot_attention()
            self.norm_input = nn.Identity()
        else:
            self.slot_attention = nn.ModuleList(
                [make_slot_attention(use_input_norm=False) for _ in range(n_blocks)]
            )
            self.norm_input = nn.LayerNorm(feature_dim)

        self.positional_embedding = positional_embedding
        self.iters = iters
        print("Iters: ", iters)
        if use_empty_slot_for_masked_slots:
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self,
        feature: ocl.typing.FeatureExtractorOutput,
        conditioning: ocl.typing.ConditioningOutput = None,
        dual_conditioning: ocl.typing.ConditioningOutput = None,
        point_conditioning: ocl.typing.ConditioningOutput = None,
        lang_conditioning: ocl.typing.ConditioningOutput = None,
        slot_mask: Optional[ocl.typing.EmptyIndicator] = None,
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply slot attention based perceptual grouping.

        Args:
            feature: Features used for grouping.
            conditioning: Initial conditioning vectors for slots.
            slot_mask: Slot mask where true indicates that the slot should be masked.

        Returns:
            The grouped features.
        """
        if self.positional_embedding:
            features = self.positional_embedding(feature.features, feature.positions)
        else:
            features = feature.features

        features = self.norm_input(features)

        if isinstance(self.slot_attention, SlotAttention):
            if conditioning is not None:
                slots, attn, _ = self.slot_attention(features, conditioning, slot_mask)
            else:
                slots, attn = None, None

            if dual_conditioning is not None:
                # lang_slots, lang_attn = self.slot_attention(features, lang_conditioning, slot_mask)
                if self.dual_conditioning:
                    dual_slots, duel_attn, dual_attn_list = self.slot_attention(
                        features, conditioning, slot_mask, dual_conditioning=dual_conditioning
                    )
                else:
                    dual_slots, dual_attn, dual_attn_list = self.slot_attention(
                        features, dual_conditioning, slot_mask
                    )
            else:
                dual_slots = None
                dual_attn = None
                dual_attn_list = None

            if point_conditioning is not None:
                if self.point_conditioning:
                    point_slots, point_attn = self.slot_attention(
                        features, conditioning, slot_mask, point_conditioning=point_conditioning
                    )
                else:
                    point_slots, point_attn = self.slot_attention(
                        features, point_conditioning, slot_mask
                    )
            else:
                point_slots = None
                point_attn = None

            if lang_conditioning is not None:
                if self.lang_conditioning:
                    lang_slots, lang_attn = self.slot_attention(
                        features, conditioning, slot_mask, lang_conditioning=lang_conditioning
                    )
                else:
                    lang_slots, lang_attn = self.slot_attention(
                        features, lang_conditioning, slot_mask
                    )
            else:
                lang_slots = None
                lang_attn = None

        else:
            slots = conditioning
            for slot_att in self.slot_attention:
                slots, attn = slot_att(features, slots, slot_mask)

            dual_slots = dual_conditioning
            for slot_att in self.slot_attention:
                dual_slots, dual_attn_attn = slot_att(features, dual_slots, slot_mask)

            point_slots = point_conditioning
            for slot_att in self.slot_attention:
                point_slots, point_attn = slot_att(features, point_slots, slot_mask)

        if slot_mask is not None and self.empty_slot is not None:
            slots[slot_mask] = self.empty_slot.to(dtype=slots.dtype)

        if dual_attn_list is None:
            return ocl.typing.ControllablePerceptualGroupingOutput(
                slots,
                dual_slots,
                point_slots,
                lang_slots,
                feature_attributions=attn,
                dual_feature_attributions=dual_attn,
                point_feature_attributions=point_attn,
                lang_feature_attributions=lang_attn,
                is_empty=slot_mask,
            )
        else:
            return ocl.typing.ControllablePerceptualGroupingOutput(
                slots,
                dual_slots,
                point_slots,
                lang_slots,
                feature_attributions=attn,
                dual_feature_attributions=dual_attn,
                point_feature_attributions=point_attn,
                lang_feature_attributions=lang_attn,
                dual_attn_1=dual_attn_list[0],
                dual_attn_2=dual_attn_list[1],
                dual_attn_3=dual_attn_list[2],
                is_empty=slot_mask,
            )


class StickBreakingGrouping(nn.Module):
    """Perceptual grouping based on a stick-breaking process.

    The idea is to pick a random feature from a yet unexplained part of the feature map, then see
    which parts of the feature map are "explained" by this feature using a kernel distance. This
    process is iterated until some termination criterion is reached. In principle, this process
    allows to extract a variable number of slots per image.

    This is based on Engelcke et al, GENESIS-V2: Inferring Unordered Object Representations without
    Iterative Refinement, http://arxiv.org/abs/2104.09958. Our implementation here differs a bit from
    the one described there:

    - It only implements one kernel distance, the Gaussian kernel
    - It does not take features positions into account when computing the kernel distances
    - It L2-normalises the input features to get comparable scales of the kernel distance
    - It has multiple termination criteria, namely termination based on fraction explained, mean
      mask value, and min-max mask value. GENESIS-V2 implements termination based on mean mask
      value, but does not mention it in the paper. Note that by default, all termination criteria
      are disabled.
    """

    def __init__(
        self,
        object_dim: int,
        feature_dim: int,
        n_slots: int,
        kernel_var: float = 1.0,
        learn_kernel_var: bool = False,
        max_unexplained: float = 0.0,
        min_slot_mask: float = 0.0,
        min_max_mask_value: float = 0.0,
        early_termination: bool = False,
        add_unexplained: bool = False,
        eps: float = 1e-8,
        detach_features: bool = False,
        use_input_layernorm: bool = False,
    ):
        """Initialize stick-breaking-based perceptual grouping.

        Args:
            object_dim: Dimensionality of extracted slots.
            feature_dim: Dimensionality of features to operate on.
            n_slots: Maximum number of slots.
            kernel_var: Variance in Gaussian kernel.
            learn_kernel_var: Whether kernel variance should be included as trainable parameter.
            max_unexplained: If fraction of unexplained features drops under this value,
                drop the slot.
            min_slot_mask: If slot mask has lower average value than this value, drop the slot.
            min_max_mask_value: If slot mask's maximum value is lower than this value,
                drop the slot.
            early_termination: If true, all slots after the first dropped slot are also dropped.
            add_unexplained: If true, add a slot that covers all unexplained parts at the point
                when the first slot was dropped.
            eps: Minimum value for masks.
            detach_features: If true, detach input features such that no gradient flows through
                this operation.
            use_input_layernorm: Apply layernorm to features prior to grouping.
        """
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        assert kernel_var > 0.0
        if learn_kernel_var:
            self.kernel_logvar = nn.Parameter(torch.tensor(math.log(kernel_var)))
        else:
            self.register_buffer("kernel_logvar", torch.tensor(math.log(kernel_var)))

        assert 0.0 <= max_unexplained < 1.0
        self.max_unexplained = max_unexplained
        assert 0.0 <= min_slot_mask < 1.0
        self.min_slot_mask = min_slot_mask
        assert 0.0 <= min_max_mask_value < 1.0
        self.min_max_mask_value = min_max_mask_value

        self.early_termination = early_termination
        self.add_unexplained = add_unexplained
        if add_unexplained and not early_termination:
            raise ValueError("`add_unexplained=True` only works with `early_termination=True`")

        self.eps = eps
        self.log_eps = math.log(eps)
        self.detach_features = detach_features

        if use_input_layernorm:
            self.in_proj = nn.Sequential(
                nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim)
            )
            torch.nn.init.xavier_uniform_(self.in_proj[-1].weight)
            torch.nn.init.zeros_(self.in_proj[-1].bias)
        else:
            self.in_proj = nn.Linear(feature_dim, feature_dim)
            torch.nn.init.xavier_uniform_(self.in_proj.weight)
            torch.nn.init.zeros_(self.in_proj.bias)

        self.out_proj = nn.Linear(feature_dim, object_dim)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, features: ocl.typing.FeatureExtractorOutput
    ) -> ocl.typing.PerceptualGroupingOutput:
        """Apply stick-breaking-based perceptual grouping to input features.

        Args:
            features: Features that should be grouped.

        Returns:
            Grouped features.
        """
        features = features.features
        bs, n_features, feature_dim = features.shape
        if self.detach_features:
            features = features.detach()

        proj_features = torch.nn.functional.normalize(self.in_proj(features), dim=-1)

        # The scope keep tracks of the unexplained parts of the feature map
        log_scope = torch.zeros_like(features[:, :, 0])
        # Seeds are used for random sampling of features
        log_seeds = torch.rand_like(log_scope).clamp_min(self.eps).log()

        slot_masks = []
        log_scopes = []

        # Always iterate for `n_iters` steps for batching reasons. Termination is modeled afterwards.
        n_iters = self.n_slots - 1 if self.add_unexplained else self.n_slots
        for _ in range(n_iters):
            log_scopes.append(log_scope)

            # Sample random features from unexplained parts of the feature map
            rand_idxs = torch.argmax(log_scope + log_seeds, dim=1)
            cur_centers = proj_features.gather(
                1, rand_idxs.view(bs, 1, 1).expand(-1, -1, feature_dim)
            )

            # Compute similarity between selected features and other features. alpha can be
            # considered an attention mask.
            dists = torch.sum((cur_centers - proj_features) ** 2, dim=-1)
            log_alpha = (-dists / self.kernel_logvar.exp()).clamp_min(self.log_eps)

            # To get the slot mask, we subtract already explained parts from alpha using the scope
            mask = (log_scope + log_alpha).exp()
            slot_masks.append(mask)

            # Update scope by masking out parts explained by the current iteration
            log_1m_alpha = (1 - log_alpha.exp()).clamp_min(self.eps).log()
            log_scope = log_scope + log_1m_alpha

        if self.add_unexplained:
            slot_masks.append(log_scope.exp())
            log_scopes.append(log_scope)

        slot_masks = torch.stack(slot_masks, dim=1)
        scopes = torch.stack(log_scopes, dim=1).exp()

        # Compute criteria for ignoring slots
        empty_slots = torch.zeros_like(slot_masks[:, :, 0], dtype=torch.bool)
        # When fraction of unexplained features drops under threshold, ignore slot,
        empty_slots |= scopes.mean(dim=-1) < self.max_unexplained
        # or when slot's mean mask is under threshold, ignore slot,
        empty_slots |= slot_masks.mean(dim=-1) < self.min_slot_mask
        # or when slot's masks maximum value is under threshold, ignore slot.
        empty_slots |= slot_masks.max(dim=-1).values < self.min_max_mask_value

        if self.early_termination:
            # Simulate early termination by marking all slots after the first empty slot as empty
            empty_slots = torch.cummax(empty_slots, dim=1).values
            if self.add_unexplained:
                # After termination, add one more slot using the unexplained parts at that point
                first_empty = torch.argmax(empty_slots.to(torch.int32), dim=1).unsqueeze(-1)
                empty_slots.scatter_(1, first_empty, torch.zeros_like(first_empty, dtype=torch.bool))

                idxs = first_empty.view(bs, 1, 1).expand(-1, -1, n_features)
                unexplained = scopes.gather(1, idxs)
                slot_masks.scatter_(1, idxs, unexplained)

        # Create slot representations as weighted average of feature map
        slots = torch.einsum("bkp,bpd->bkd", slot_masks, features)
        slots = slots / slot_masks.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        slots = self.out_proj(slots)

        # Zero-out masked slots
        slots.masked_fill_(empty_slots.view(bs, slots.shape[1], 1), 0.0)

        return ocl.typing.PerceptualGroupingOutput(
            slots, feature_attributions=slot_masks, is_empty=empty_slots
        )


class KMeansGrouping(nn.Module):
    """Simple K-means clustering based grouping."""

    def __init__(
        self,
        n_slots: int,
        use_l2_normalization: bool = True,
        clustering_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._object_dim = None
        self.n_slots = n_slots
        self.use_l2_normalization = use_l2_normalization

        kwargs = clustering_kwargs if clustering_kwargs is not None else {}
        self.make_clustering = lambda: cluster.KMeans(n_clusters=n_slots, **kwargs)

    @property
    def object_dim(self):
        return self._object_dim

    def forward(
        self, feature: ocl.typing.FeatureExtractorOutput
    ) -> ocl.typing.PerceptualGroupingOutput:
        feature = feature.features
        if self._object_dim is None:
            self._object_dim = feature.shape[-1]

        if self.use_l2_normalization:
            feature = torch.nn.functional.normalize(feature, dim=-1)

        batch_features = feature.detach().cpu().numpy()

        cluster_ids = []
        cluster_centers = []

        for feat in batch_features:
            clustering = self.make_clustering()

            cluster_ids.append(clustering.fit_predict(feat).astype(numpy.int64))
            cluster_centers.append(clustering.cluster_centers_)

        cluster_ids = torch.from_numpy(numpy.stack(cluster_ids))
        cluster_centers = torch.from_numpy(numpy.stack(cluster_centers))

        slot_masks = torch.nn.functional.one_hot(cluster_ids, num_classes=self.n_slots)
        slot_masks = slot_masks.transpose(-2, -1).to(torch.float32)

        return ocl.typing.PerceptualGroupingOutput(
            cluster_centers.to(feature.device), feature_attributions=slot_masks.to(feature.device)
        )
