"""Registered RoboLab task catalog.

Snapshot of `robolab.registrations.droid_jointpos.auto_env_registrations`'s
output as of 2026-04-11. Regenerate with:

    OMNI_KIT_ACCEPT_EULA=YES uv run python \
        ../../third_party/robolab/scripts/check_registered_envs.py

Importing this module does NOT boot Isaac Sim — it's a pure-Python constant,
safe to import from eval_all.py or any tooling that needs to enumerate tasks
without paying the 30-second AppLauncher cost.

``TASKS`` maps each task name to the set of tags RoboLab assigns it. Use
``tasks_with_tag("simple")`` to filter by tag, or just iterate ``TASKS.keys()``
to run everything.
"""

from __future__ import annotations

TASKS: dict[str, frozenset[str]] = {
    "ReorientAllMugsTask": frozenset({"all", "complex", "reorientation"}),
    "ButterAboveRaisinTask": frozenset({"all", "simple", "spatial"}),
    "UtensilsInMugTask": frozenset({"affordance", "all", "moderate", "semantics"}),
    "GrabAFruitTask": frozenset({"all", "semantics", "simple"}),
    "FruitsGreenLimesOnPlateTask": frozenset({"all", "color", "simple"}),
    "BananaThenRubiksCubeTask": frozenset({"all", "conjunction", "simple"}),
    "PickGlassesTask": frozenset({"all", "semantics", "simple"}),
    "FruitsOnionToPlateTask": frozenset({"all", "semantics", "simple", "vague"}),
    "BananaOnPlateTask": frozenset({"all", "simple"}),
    "RubiksCubeRightOfBowlTask": frozenset({"all", "moderate", "spatial"}),
    "ClutterPlasticTask": frozenset({"all", "moderate", "semantics"}),
    "RecycleCartonTask": frozenset({"all", "semantics", "simple", "vague"}),
    "NonHammerToolsInRightBinTask": frozenset(
        {"all", "moderate", "semantics", "sorting", "spatial"}
    ),
    "MoveBananaToBagelPlateTask": frozenset({"all", "semantics", "simple"}),
    "RecycleCartonsOnBoxTask": frozenset({"all", "semantics", "simple"}),
    "ThrowAwaySnacksTask": frozenset({"all", "semantics", "simple"}),
    "SmallerObjectButterInBinTask": frozenset({"all", "simple", "size"}),
    "RubiksCubeLeftOfBowlTask": frozenset({"all", "moderate", "spatial"}),
    "MouseOnKeyboardTask": frozenset({"all", "semantics", "simple"}),
    "BananaInBowlTask": frozenset({"all", "semantics", "simple"}),
    "BagelsOnPlateTask": frozenset({"all", "semantics", "simple"}),
    "SpoonInMugTask": frozenset({"affordance", "all", "moderate", "spatial"}),
    "ElectronicsInBinTask": frozenset({"all", "complex", "semantics", "sorting"}),
    "YogurtInBowlTask": frozenset({"all", "color", "simple", "size"}),
    "Stack3RubiksCubeTask": frozenset({"all", "moderate", "stacking"}),
    "PutMugsOnShelfTask": frozenset(
        {"affordance", "all", "counting", "moderate", "spatial"}
    ),
    "GrabABagelTask": frozenset({"all", "semantics", "simple"}),
    "RubiksCubeBehindBowlTask": frozenset({"all", "moderate", "spatial"}),
    "ToolOrganizationBothTask": frozenset({"all", "moderate", "semantics", "spatial"}),
    "RubiksCubesInBinTask": frozenset({"all", "complex", "sorting"}),
    "PickOrangeObjectTask": frozenset({"all", "color", "simple"}),
    "RubiksCubeThenBananaTask": frozenset({"all", "conjunction", "simple"}),
    "CoffeePotInBinTask": frozenset({"all", "semantics", "simple"}),
    "CleanUpToysTask": frozenset({"all", "complex", "semantics", "sorting"}),
    "OneBottleOnShelfTask": frozenset({"all", "semantics", "simple"}),
    "RubiksCubeTask": frozenset({"all", "simple"}),
    "ReorientRedMugTask": frozenset({"all", "color", "moderate", "reorientation"}),
    "OneBottleInSquarePailTask": frozenset({"all", "semantics", "simple"}),
    "BananasOutOfBinTask": frozenset({"all", "moderate", "semantics", "spatial"}),
    "ReorientWhiteMugsTask": frozenset({"all", "color", "moderate", "reorientation"}),
    "PinkSpoonInPotTask": frozenset({"affordance", "all", "color", "moderate"}),
    "RubiksCubeInFrontOfBowlTask": frozenset({"all", "moderate", "spatial"}),
    "PutBowlOnShelfTopTask": frozenset({"all", "simple", "spatial"}),
    "RubiksCubeAndBananaTask": frozenset({"all", "conjunction", "simple"}),
    "PickDrillTask": frozenset({"all", "semantics", "simple"}),
    "CannedFoodInBinTask": frozenset({"all", "semantics", "simple"}),
    "TakeSpatulaOffShelfTask": frozenset({"affordance", "all", "moderate", "spatial"}),
    "WhiteMugsInBinTask": frozenset({"all", "color", "simple"}),
    "TakeMeasuringSpoonOutTask": frozenset({"all", "semantics", "simple"}),
    "FruitsOnPlateTask": frozenset({"all", "complex", "semantics", "vague"}),
    "YellowAndWhiteObjectsInBinTask": frozenset(
        {"all", "color", "conjunction", "simple"}
    ),
    "CookingPickPastaToolTask": frozenset(
        {"all", "color", "simple", "spatial", "vague"}
    ),
    "ReorientJugTask": frozenset(
        {"affordance", "all", "moderate", "reorientation", "semantics"}
    ),
    "ClutterPumpkinTask": frozenset({"all", "semantics", "simple"}),
    "ToyInBinTask": frozenset({"all", "semantics", "simple"}),
    "BlockStackingSpecifiedOrderTask": frozenset(
        {"all", "color", "complex", "stacking"}
    ),
    "BlockStackingOrderAgnosticTask": frozenset({"all", "complex", "stacking"}),
    "PutTwoMugsOnShelfTask": frozenset(
        {"affordance", "all", "counting", "moderate", "spatial"}
    ),
    "SauceBottlesCrateTask": frozenset({"all", "color", "semantics", "simple"}),
    "BBQSauceInBinTask": frozenset({"all", "color", "semantics", "simple"}),
    "FoodPacking1BoxesTask": frozenset({"all", "semantics", "simple"}),
    "FruitsOnionTask": frozenset({"all", "semantics", "simple"}),
    "BlackItemsInBinTask": frozenset({"all", "color", "complex", "sorting"}),
    "AppleAndYogurtInBowlTask": frozenset({"affordance", "all", "moderate"}),
    "ThrowAwayAppleTask": frozenset({"all", "semantics", "simple"}),
    "FoodPacking2CansTask": frozenset({"all", "semantics", "simple"}),
    "ToolsPickingHammerTask": frozenset(
        {"all", "color", "semantics", "simple", "spatial"}
    ),
    "SmallPumpkinInBinTask": frozenset({"all", "simple", "size"}),
    "PickUpBluePitcherTask": frozenset({"affordance", "all", "color", "moderate"}),
    "RedDishesInBinTask": frozenset({"all", "color", "semantics", "simple"}),
    "DishesInBinTask": frozenset({"all", "moderate", "semantics", "vague"}),
    "FoodPackingByColorTask": frozenset(
        {"all", "color", "moderate", "sorting", "spatial"}
    ),
    "MustardInLeftBinTask": frozenset({"all", "simple", "spatial"}),
    "BowlStackingLeftOnRightTask": frozenset({"all", "simple", "spatial"}),
    "RedItemsInBinTask": frozenset({"all", "color", "moderate", "sorting"}),
    "FoodPacking3CansTask": frozenset({"all", "moderate", "semantics"}),
    "GreenSpoonsInPotTask": frozenset({"all", "color", "complex", "reorientation"}),
    "ToolOrganizationTask": frozenset(
        {"all", "color", "moderate", "semantics", "spatial"}
    ),
    "PickUpGreenObjectTask": frozenset({"all", "color", "simple"}),
    "PhoneOrRemoteInBinTask": frozenset({"all", "conjunction", "simple"}),
    "UnstackRubiksCubeTask": frozenset({"all", "moderate", "stacking"}),
    "ToolsPickingDrillTask": frozenset({"all", "semantics", "simple", "spatial"}),
    "FoodPacking3BoxesTask": frozenset({"all", "moderate", "semantics"}),
    "BigPumpkinInBinTask": frozenset({"all", "simple", "size"}),
    "BowlStackingRightOnLeftTask": frozenset({"all", "simple", "spatial"}),
    "FruitsMovingTask": frozenset({"all", "color", "simple"}),
    "LargerObjectRaisinBoxInBinTask": frozenset({"all", "simple", "size"}),
    "ToolsPickingAllHammersTask": frozenset({"all", "complex", "semantics", "spatial"}),
    "FruitsMovingOrangeOrLimeTask": frozenset({"all", "conjunction", "simple"}),
    "BlocksInBinTask": frozenset({"all", "complex", "sorting"}),
    "MustardInRightBinTask": frozenset({"all", "simple", "spatial"}),
    "RubiksCubeOrBananaTask": frozenset({"all", "conjunction", "simple"}),
    "AnimalsInBinTask": frozenset({"all", "semantics", "simple"}),
    "MarkerInMugTask": frozenset({"affordance", "all", "moderate"}),
    "SmartphoneInBinTask": frozenset({"all", "semantics", "simple"}),
    "ClampInRightBinTask": frozenset({"all", "semantics", "simple", "spatial"}),
    "CookingClearPlateTask": frozenset({"all", "color", "moderate", "sorting"}),
    "MustardAboveRaisinTask": frozenset({"all", "simple", "spatial"}),
    "StackYellowOnRedTask": frozenset({"all", "moderate", "stacking"}),
    "FruitsOrangesOnPlateTask": frozenset(
        {"all", "counting", "moderate", "semantics", "vague"}
    ),
    "TakeMugsOffOfShelfTask": frozenset({"affordance", "all", "moderate", "semantics"}),
    "SpoonsInPotTask": frozenset(
        {"affordance", "all", "complex", "reorientation", "semantics"}
    ),
    "FoodPacking2BoxesTask": frozenset({"all", "semantics", "simple"}),
    "WhiteMugInCenterOfTableTask": frozenset({"all", "color", "moderate", "spatial"}),
    "StackWhiteMugsTask": frozenset({"all", "color", "moderate", "stacking"}),
    "KeyboardOutOfBinTask": frozenset({"all", "simple", "spatial"}),
    "WoodSpatulaToBowlTask": frozenset({"all", "semantics", "simple"}),
    "BananasInBinThreeTotalTask": frozenset(
        {"all", "counting", "moderate", "semantics"}
    ),
    "BananasInBinOneMoreTask": frozenset({"all", "counting", "moderate", "semantics"}),
    "FoodPacking1CansTask": frozenset({"all", "semantics", "simple"}),
    "RecycleCartonsVerticalCrateTask": frozenset(
        {"all", "moderate", "semantics", "spatial"}
    ),
    "HammersInLeftBinTask": frozenset(
        {"all", "color", "moderate", "semantics", "spatial"}
    ),
    "FruitsOnPlate3Task": frozenset(
        {"all", "complex", "counting", "semantics", "vague"}
    ),
    "ClearOrganicObjectsTask": frozenset({"all", "complex", "semantics"}),
    "CondimentsInBinTask": frozenset({"all", "complex", "semantics", "sorting"}),
    "JugsOnShelfTask": frozenset({"all", "semantics", "simple"}),
    "BowlInBinTask": frozenset({"all", "simple"}),
    "PlasticBottlesInSquarePailTask": frozenset(
        {"all", "complex", "semantics", "size", "sorting"}
    ),
    "BananasInCrateTask": frozenset({"all", "counting", "moderate"}),
    "CubesAndBlocksInBinTask": frozenset({"all", "complex", "conjunction", "sorting"}),
}

ALL_TAGS: frozenset[str] = frozenset().union(*TASKS.values())


def tasks_with_tag(tag: str) -> list[str]:
    """Return the names of all tasks carrying ``tag``, sorted for determinism."""
    return sorted(name for name, tags in TASKS.items() if tag in tags)


def tasks_with_all_tags(*tags: str) -> list[str]:
    """Return the names of tasks that carry every tag in ``tags``."""
    required = frozenset(tags)
    return sorted(name for name, task_tags in TASKS.items() if required <= task_tags)
