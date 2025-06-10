"""Conversion script for ClevrTex dataset.

ClevrTex was introduced by Karazija et al, 2021 in "ClevrTex: A Texture-Rich Benchmark for
Unsupervised Multi-Object Segmentation". See https://www.robots.ox.ac.uk/~vgg/data/clevrtex/.

By default, this script writes out center crop images and masks at full resolution (240px), which
is different from the ClevrTex paper/dataloader, where they use a 0.8x zoomed-in center crop and
resolution 128px.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import webdataset
from PIL import Image
from utils import get_shard_pattern

logging.getLogger().setLevel(logging.INFO)

SIZES = ["large", "medium", "small"]
SHAPES = ["cube", "cylinder", "monkey", "sphere"]
SHAPES_OUTD = ["cone", "icosahedron", "teapot", "torus"]
COLORS_VBG = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
MATERIALS = [
    "PoliigonBricks01",
    "PoliigonBricksFlemishRed001",
    "PoliigonBricksPaintedWhite001",
    "PoliigonCarpetTwistNatural001",
    "PoliigonChainmailCopperRoundedThin001",
    "PoliigonCityStreetAsphaltGenericCracked002",
    "PoliigonCityStreetRoadAsphaltTwoLaneWorn001",
    "PoliigonCliffJagged004",
    "PoliigonCobblestoneArches002",
    "PoliigonConcreteWall001",
    "PoliigonFabricDenim003",
    "PoliigonFabricFleece001",
    "PoliigonFabricLeatherBuffaloRustic001",
    "PoliigonFabricRope001",
    "PoliigonFabricUpholsteryBrightAnglePattern001",
    "PoliigonGroundClay002",
    "PoliigonGroundDirtForest014",
    "PoliigonGroundDirtRocky002",
    "PoliigonGroundForest003",
    "PoliigonGroundForest008",
    "PoliigonGroundForestMulch001",
    "PoliigonGroundForestRoots001",
    "PoliigonGroundMoss001",
    "PoliigonGroundSnowPitted003",
    "PoliigonGroundTireTracks001",
    "PoliigonInteriorDesignRugStarryNight001",
    "PoliigonMarble062",
    "PoliigonMarble13",
    "PoliigonMetalCorrodedHeavy001",
    "PoliigonMetalCorrugatedIronSheet002",
    "PoliigonMetalDesignerWeaveSteel002",
    "PoliigonMetalPanelRectangular001",
    "PoliigonMetalSpottyDiscoloration001",
    "PoliigonMetalStainlessSteelBrushed",
    "PoliigonPlaster07",
    "PoliigonPlaster17",
    "PoliigonRoadCityWorn001",
    "PoliigonRoofTilesTerracotta004",
    "PoliigonRustMixedOnPaint012",
    "PoliigonRustPlain007",
    "PoliigonSolarPanelsPolycrystallineTypeBFramedClean001",
    "PoliigonStoneBricksBeige015",
    "PoliigonStoneMarbleCalacatta004",
    "PoliigonTerrazzoVenetianMatteWhite001",
    "PoliigonTiles05",
    "PoliigonTilesMarbleChevronCreamGrey001",
    "PoliigonTilesMarbleSageGreenBrickBondHoned001",
    "PoliigonTilesOnyxOpaloBlack001",
    "PoliigonTilesRectangularMirrorGray001",
    "PoliigonWallMedieval003",
    "PoliigonWaterDropletsMixedBubbled001",
    "PoliigonWoodFineDark004",
    "PoliigonWoodFlooring044",
    "PoliigonWoodFlooring061",
    "PoliigonWoodFlooringMahoganyAfricanSanded001",
    "PoliigonWoodFlooringMerbauBrickBondNatural001",
    "PoliigonWoodPlanks028",
    "PoliigonWoodPlanksWorn33",
    "PoliigonWoodQuarteredChiffon001",
    "WhiteMarble",
]
MATERIALS_OUTD = [
    "PoliigonBrickOldRed001",
    "PoliigonCarpetPlushDesignerRhombus001",
    "PoliigonChainmailGoldRounded001",
    "PoliigonChippedPaint015",
    "PoliigonCliffJagged007",
    "PoliigonDrywallPrepared007",
    "PoliigonFabricUpholsteryPyramidsPattern001",
    "PoliigonGroundClay006",
    "PoliigonGroundClay008",
    "PoliigonGroundDirtForestRoots004",
    "PoliigonGroundDirtRocky006",
    "PoliigonGroundGrassGreen",
    "PoliigonGroundRocky006",
    "PoliigonGroundRocky014",
    "PoliigonGroundSandFootprintsLeaves001",
    "PoliigonMetalSiding004",
    "PoliigonMetalStainlessSteelBraided001",
    "PoliigonPanelsAcousticFoamTiles004",
    "PoliigonPlasterSloppy",
    "PoliigonRoofTilesCeramic001",
    "PoliigonRoofTilesCopper003",
    "PoliigonRoofTilesQuarry001",
    "PoliigonRustPlain030",
    "PoliigonSteelFloorDiamond001",
    "PoliigonTilesTerrazzoPalladianaHonedGreen001",
]
MATERIALS_VBG = ["MyMetal", "Rubber"]
BG_MATERIALS_PBG = ["TabulaRasa"]

MATERIALS_BY_VARIANT = {
    "full": MATERIALS,
    "pbg": MATERIALS,
    "vbg": MATERIALS_VBG,
    "grassbg": MATERIALS,
    "camo": MATERIALS,
    "outd": MATERIALS_OUTD,
}
BG_MATERIALS_BY_VARIANT = {
    "full": MATERIALS,
    "pbg": BG_MATERIALS_PBG,
    "vbg": MATERIALS,
    "grassbg": MATERIALS,
    "camo": MATERIALS,
    "outd": MATERIALS_OUTD,
}


class ClevrTex:
    """Official ClevrTex loader adapted from https://github.com/karazijal/clevrtex-generation.

    Released under BSD 3 license.
    """

    ccrop_frac = 0.8
    splits = {"test": (0.0, 0.1), "val": (0.1, 0.2), "train": (0.2, 1.0)}
    shape = (3, 240, 320)
    variants = {"full", "pbg", "vbg", "grassbg", "camo", "outd"}

    def __init__(
        self,
        path: Path,
        dataset_variant="full",
        split="train",
        crop=True,
        crop_factor=0.8,
        resize=(128, 128),
        return_metadata=True,
    ):
        self.return_metadata = return_metadata
        self.crop = crop
        self.crop_factor = crop_factor
        self.resize = resize
        if dataset_variant not in self.variants:
            raise ValueError(
                f"Unknown variant {dataset_variant}; [{', '.join(self.variants)}] available "
            )

        if split not in self.splits:
            raise ValueError(f"Unknown split {split}; [{', '.join(self.splits)}] available ")
        if dataset_variant == "outd":
            # No dataset splits in
            split = None

        self.dataset_variant = dataset_variant
        self.split = split

        self.basepath = Path(path)
        if not self.basepath.exists():
            raise ValueError()
        sub_fold = self._variant_subfolder()
        if self.basepath.name != sub_fold:
            self.basepath = self.basepath / sub_fold
        self.index, self.mask_index, self.metadata_index = self._reindex()

        print(f"Sourced {dataset_variant} ({split}) from {self.basepath}")

        bias, limit = self.splits.get(split, (0.0, 1.0))
        if isinstance(bias, float):
            bias = int(bias * len(self.index))
        if isinstance(limit, float):
            limit = int(limit * len(self.index))
        self.limit = limit
        self.bias = bias

    def _index_with_bias_and_limit(self, idx):
        if idx >= 0:
            idx += self.bias
            if idx >= self.limit:
                raise IndexError()
        else:
            idx = self.limit + idx
            if idx < self.bias:
                raise IndexError()
        return idx

    def _reindex(self):
        print(f"Indexing {self.basepath}")

        img_index = {}
        msk_index = {}
        met_index = {}

        prefix = f"CLEVRTEX_{self.dataset_variant}_"

        img_suffix = ".png"
        msk_suffix = "_flat.png"
        met_suffix = ".json"

        _max = 0
        for img_path in self.basepath.glob(f"**/{prefix}??????{img_suffix}"):
            indstr = img_path.name.replace(prefix, "").replace(img_suffix, "")
            msk_path = img_path.parent / f"{prefix}{indstr}{msk_suffix}"
            met_path = img_path.parent / f"{prefix}{indstr}{met_suffix}"
            indstr_stripped = indstr.lstrip("0")
            if indstr_stripped:
                ind = int(indstr)
            else:
                ind = 0
            if ind > _max:
                _max = ind

            if not msk_path.exists():
                raise ValueError(f"Missing {msk_suffix.name}")

            if ind in img_index:
                raise ValueError(f"Duplica {ind}")

            img_index[ind] = img_path
            msk_index[ind] = msk_path
            if self.return_metadata:
                if not met_path.exists():
                    raise ValueError(f"Missing {met_path.name}")
                met_index[ind] = met_path
            else:
                met_index[ind] = None

        if len(img_index) == 0:
            raise ValueError("No values found")
        missing = [i for i in range(0, _max) if i not in img_index]
        if missing:
            raise ValueError(f"Missing images numbers {missing}")

        return img_index, msk_index, met_index

    def _variant_subfolder(self):
        return f"clevrtex_{self.dataset_variant.lower()}"

    def _format_metadata(self, meta):
        """Drop unimportanat, unused or incorrect data from metadata.

        Data may become incorrect due to transformations,
        such as cropping and resizing would make pixel coordinates incorrect.
        Furthermore, only VBG dataset has color assigned to objects, we delete the value for others.
        """
        objs = []
        for obj in meta["objects"]:
            o = {
                "material": obj["material"],
                "shape": obj["shape"],
                "size": obj["size"],
                "rotation": obj["rotation"],
            }
            if self.dataset_variant == "vbg":
                o["color"] = obj["color"]
            objs.append(o)
        return {"ground_material": meta["ground_material"], "objects": objs}

    def __len__(self):
        return self.limit - self.bias

    def __getitem__(self, ind):
        ind = self._index_with_bias_and_limit(ind)

        img = Image.open(self.index[ind])
        msk = Image.open(self.mask_index[ind])

        if self.crop:
            crop_size = int(self.crop_factor * float(min(img.width, img.height)))
            img = img.crop(
                (
                    (img.width - crop_size) // 2,
                    (img.height - crop_size) // 2,
                    (img.width + crop_size) // 2,
                    (img.height + crop_size) // 2,
                )
            )
            msk = msk.crop(
                (
                    (msk.width - crop_size) // 2,
                    (msk.height - crop_size) // 2,
                    (msk.width + crop_size) // 2,
                    (msk.height + crop_size) // 2,
                )
            )
        if self.resize and img.size != self.resize:
            img = img.resize(self.resize, resample=Image.BILINEAR)
            msk = msk.resize(self.resize, resample=Image.NEAREST)

        # Masks are stored in format "P" (palette). Convert to "L" (grayscale) using a numpy array
        # in between to forget about the palette but preserve the integer-based mask indexing.
        msk = Image.fromarray(np.array(msk), mode="L")

        ret = (ind, img, msk)

        if self.return_metadata:
            with self.metadata_index[ind].open("r") as inf:
                meta = json.load(inf)
            ret = (ind, img, msk, self._format_metadata(meta))

        return ret


def get_visibilities(mask: np.ndarray, max_num_objects: int = 10) -> np.array:
    visibilities = []
    for idx in range(1, max_num_objects + 1):
        visibilities.append(np.any(mask == idx))
    return np.array(visibilities, dtype=bool)


def get_materials(materials: List[str], variant: str) -> np.array:
    source = [m.lower() for m in MATERIALS_BY_VARIANT[variant]]
    return np.array([source.index(m) for m in materials], dtype=np.uint8)


def get_bg_material(bg_material: str, variant: str) -> np.array:
    source = [m.lower() for m in BG_MATERIALS_BY_VARIANT[variant]]
    return np.array([source.index(bg_material)], dtype=np.uint8)


def get_shapes(shapes: List[str], variant: str) -> np.array:
    source = SHAPES if variant != "outd" else SHAPES_OUTD
    return np.array([source.index(m) for m in shapes], dtype=np.uint8)


def get_sizes(sizes: List[str]) -> np.array:
    return np.array([SIZES.index(m) for m in sizes], dtype=np.uint8)


def get_colors(colors: List[str], variant) -> np.array:
    assert variant == "vbg"
    return np.array([COLORS_VBG.index(m) for m in colors], dtype=np.uint8)


def main(data_dir: Path, output_dir: Path, variant: str, split: str, dataset_kwargs: Dict[str, Any]):
    if split is None and variant != "outd":
        split = "train"

    output_subdir = f"{variant}_{split}" if variant != "outd" else variant
    output_path = output_dir / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(
        f"Creating webdataset for ClevrTex variant {variant}, split {split} at {output_path}"
    )
    dataset = ClevrTex(data_dir, variant, split, **dataset_kwargs)

    # Setup parameters for shard writers.
    shard_writer_params = {
        "maxsize": 100 * 1024 * 1024,  # 100 MB
        "maxcount": 5000,
        "keep_meta": True,
    }
    instance_count = 0

    # Create shards of data.
    with webdataset.ShardWriter(get_shard_pattern(output_path), **shard_writer_params) as writer:
        for idx, image, mask, meta in dataset:
            output = {}
            output["__key__"] = str(idx)
            output["image.png"] = image
            output["mask.png"] = mask
            output["visibilities.npy"] = get_visibilities(np.array(mask))
            output["materials.npy"] = get_materials(
                [m["material"] for m in meta["objects"]], variant
            )
            output["bg_material.npy"] = get_bg_material(meta["ground_material"], variant)
            output["shapes.npy"] = get_shapes([m["shape"] for m in meta["objects"]], variant)
            output["sizes.npy"] = get_sizes([m["size"] for m in meta["objects"]])
            if variant == "vbg":
                output["colors.npy"] = get_colors([m["color"] for m in meta["objects"]], variant)
            # TODO: write out object positions (after cropping and rescaling)

            writer.write(output)
            instance_count += 1

    logging.info(f"Wrote {instance_count} instances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=ClevrTex.variants,
        help=(
            "Variant to use: normal (full), pbg (plain backgrounds), "
            "vbg (varied backgrounds), grassbg (grass backgrounds), camo (camouflage), "
            "outd (OOD)"
        ),
    )
    parser.add_argument("--split", type=str, choices=list(ClevrTex.splits.keys()))
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=(240, 240),
        help="(width, height) to resize images to. Original resolution is 320x240",
    )
    parser.add_argument("--no-center-crop", action="store_true", default=False)
    parser.add_argument("--crop-factor", type=float, default=1.0)

    args = parser.parse_args()
    main(
        args.data_dir,
        args.output_dir,
        args.variant,
        args.split,
        dict(resize=args.size, crop=not args.no_center_crop, crop_factor=args.crop_factor),
    )
