"""Scenario and runtime constants for the combat arena."""

MAP_NAME = "Flat128"
CENTER = (64.0, 64.0)

MARINE_COUNT = 12
BANELING_COUNT = 6
ZERGLING_COUNT = 10
MARINE_OFFSET = (-7.0, 0.0)
BANELING_OFFSET = (7.0, 0.0)
ZERGLING_OFFSET = (8.0, 2.0)

LOOPS_PER_SECOND = 22.4
DECISION_INTERVAL_LOOPS = 4
ENEMY_REORDER_INTERVAL_LOOPS = 8
MAX_FIGHT_LOOPS = 1344
SPAWN_WAIT_LOOPS = 96
DECISION_BUDGET_MS = DECISION_INTERVAL_LOOPS / LOOPS_PER_SECOND * 1000

STIM_MIN_HP = 20
SPREAD_STEP = 2.0
RETREAT_DISTANCE = 4.0

PRICE_INPUT_PER_MTOK = 0.042
PRICE_OUTPUT_PER_MTOK = 0.0

ACTIONS = {
    "spread": "Split marines apart so baneling splash hits fewer of them",
    "clump": "Group marines tightly to focus fire",
    "retreat": "Move away from the enemy to buy time and kite",
    "attack": "Attack-move toward the enemy",
    "stim": "Use stimpack: faster movement and attack, costs 10 HP",
}

SQUAD_INSTRUCTIONS = (
    "You control a squad of Terran Marines fighting Zerg Banelings and Zerglings. "
    "Pick the best squad action for the next 0.2 seconds."
)

STATE_RULES = (
    "Banelings explode on contact and deal splash damage to every marine nearby; "
    "clumped marines die together. Marines have range 5 and outrange zerglings. "
    "Stim is strong but costs 10 HP per marine."
)
