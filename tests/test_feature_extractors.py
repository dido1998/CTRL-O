import math

import pytest
import torch

from ocl import feature_extractors


@pytest.mark.parametrize(
    "model_name,feature_level,aux_feature_levels,inp_size,outp_size,dyn_img_size,dyn_img_pad",
    [
        ("resnet18", 4, None, 224, 7, False, False),
        ("resnet34_savi", 4, None, 128, 16, False, False),
        ("vit_tiny_patch16_224", None, None, 224, 14, False, False),
        ("vit_tiny_patch16_224", 2, None, 224, 14, False, False),
        ("vit_tiny_patch16_224", (2, 12), ["block1", "block2"], 224, 14, False, False),
        ("vit_tiny_patch16_224", "block1", None, 224, 14, False, False),
        ("vit_tiny_patch16_224", "key1", None, 224, 14, False, False),
        ("vit_tiny_patch16_224", "value1", None, 224, 14, False, False),
        ("vit_tiny_patch16_224", None, ["query1"], 224, 14, False, False),
        ("vit_tiny_patch16_224", "query1", ["query1"], 224, 14, False, False),
        ("vit_small_patch14_dinov2", None, None, 518, 37, False, False),
        ("vit_small_patch14_dinov2", None, None, 112, 8, True, False),
        ("vit_small_patch14_reg4_dinov2", None, None, 113, 9, True, True),
    ],
)
def test_timm_feature_extractors(
    model_name,
    feature_level,
    aux_feature_levels,
    inp_size,
    outp_size,
    dyn_img_size,
    dyn_img_pad,
):
    extractor = feature_extractors.TimmFeatureExtractor(
        model_name,
        feature_level=feature_level,
        aux_features=aux_feature_levels,
        pretrained=False,
        freeze=True,
        dynamic_img_size=dyn_img_size,
        dynamic_img_pad=dyn_img_pad,
    )

    bs = 2
    image = torch.rand(bs, 3, inp_size, inp_size)
    features, _, aux_features = extractor.forward_images(image)
    assert features.shape[0] == bs
    assert features.shape[1] == outp_size**2
    assert features.shape[2] == extractor.feature_dim

    if aux_feature_levels is None:
        assert aux_features is None
    else:
        assert list(aux_features.keys()) == aux_feature_levels


@pytest.mark.parametrize(
    "model_name,feature_level,freeze,n_blocks_to_unfreeze",
    [
        ("resnet18", 2, False, 0),
        ("resnet18", 2, True, 0),
        ("vit_tiny_patch16_224", None, False, 0),
        ("vit_tiny_patch16_224", None, True, 0),
        ("vit_tiny_patch16_224", None, True, 4),
    ],
)
def test_timm_feature_extractors_freeze(model_name, feature_level, freeze, n_blocks_to_unfreeze):
    extractor = feature_extractors.TimmFeatureExtractor(
        model_name,
        feature_level=feature_level,
        pretrained=False,
        freeze=freeze,
        n_blocks_to_unfreeze=n_blocks_to_unfreeze,
    )

    bs = 2
    image = torch.rand(bs, 3, 224, 224)
    features, *_ = extractor.forward_images(image)

    loss = features.mean()

    if freeze and n_blocks_to_unfreeze == 0:
        assert loss.grad_fn is None
    else:
        loss.backward()
        if n_blocks_to_unfreeze == 0:
            for param in extractor.parameters():
                assert param.grad is not None
        else:
            if extractor.is_vit:
                for param in extractor.model.blocks[:-n_blocks_to_unfreeze].parameters():
                    assert param.grad is None
                for param in extractor.model.blocks[-n_blocks_to_unfreeze:].parameters():
                    assert param.grad is not None


@pytest.mark.parametrize("masking_rate", [0, 0.8])
def test_masked_vision_transformer(masking_rate):
    extractor = feature_extractors.TimmFeatureExtractor(
        model_name="masked_vit_small_patch16_224.dino",
        feature_level=1,
        pretrained=False,
        masking_rate=masking_rate,
    )

    bs, n_tokens, feature_dim = 2, 196, 384
    image = torch.rand(bs, 3, 224, 224)
    features, positions, aux_features = extractor.forward_images(image)

    n_output_tokens = math.ceil(n_tokens * (1 - masking_rate))

    assert features.shape == (bs, n_output_tokens, feature_dim)
    assert positions.shape == (bs, n_output_tokens, 2)
    assert aux_features["mask"].shape == (bs, n_tokens)
    assert aux_features["mask_indices_keep"].shape == (bs, n_output_tokens)
    assert aux_features["mask_indices_restore"].shape == (bs, n_tokens)
    assert torch.all(aux_features["mask"].sum(dim=1) == n_tokens - n_output_tokens)
