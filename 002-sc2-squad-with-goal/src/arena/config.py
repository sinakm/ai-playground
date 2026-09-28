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
PRE_FIGHT_WAIT_LOOPS = 448
CAMERA_FOLLOW_INTERVAL_LOOPS = 12
DECISION_BUDGET_MS = DECISION_INTERVAL_LOOPS / LOOPS_PER_SECOND * 1000

STIM_MIN_HP = 20
SPREAD_STEP = 2.0
RETREAT_DISTANCE = 4.0
SPLASH_DANGER_RADIUS = 3.0

PRICE_INPUT_PER_MTOK = 0.042
PRICE_OUTPUT_PER_MTOK = 0.0

ACTIONS = {
    "spread": "Step apart away from banelings, then keep attacking (kiting)",
    "clump": "Group marines tightly to focus fire",
    "retreat": "Move away from the enemy to buy time and kite",
    "attack": "Attack-move toward the enemy",
    "stim": "Use stimpack: faster movement and attack, costs 10 HP",
}

SQUAD_INSTRUCTIONS = (
    "You control a squad of 12 Terran Marines fighting Zerg Banelings and Zerglings. "
    "Goal: kill as many Zerg as possible while keeping at least one Marine alive. "
    "Score = Zerg killed if any Marine survives, otherwise 0. "
    "Running out the 60 second clock is a failure. "
    "Marines only deal damage while attacking. "
    "Pick the squad action for the next 0.2 seconds that best serves this goal."
)

STATE_RULES = (
    "Banelings explode on contact and deal splash damage to every marine nearby; "
    "clumped marines die together. Marines have range 5 and outrange zerglings. "
    "Stim is strong but costs 10 HP per marine. "
    "Marines do not shoot while moving. The spread action steps away and then keeps attacking."
)
