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
DECISION_INTERVAL_LOOPS = 12
ENEMY_REORDER_INTERVAL_LOOPS = 8
MAX_FIGHT_LOOPS = 1344
SPAWN_WAIT_LOOPS = 96
PRE_FIGHT_WAIT_LOOPS = 448
CAMERA_FOLLOW_INTERVAL_LOOPS = 12
DECISION_BUDGET_MS = DECISION_INTERVAL_LOOPS / LOOPS_PER_SECOND * 1000

STIM_MIN_HP = 20
SPREAD_STEP = 1.5
RETREAT_DISTANCE = 4.0
SPLASH_DANGER_RADIUS = 3.0

PRICE_INPUT_PER_MTOK = 0.042
PRICE_OUTPUT_PER_MTOK = 0.0

ACTIONS = {
    "spread": "Banelings are within 3 cells of the squad and marines are clumped: step apart, then keep attacking",
    "clump": "No banelings remain, only zerglings: group up and focus fire",
    "retreat": "Banelings are about to reach a clumped squad and marines cannot spread in time: back off while they chase",
    "attack": "No baneling is within 4 cells: attack the enemy now",
    "stim": "Enemies are within 6 cells, no marine is stimmed and most marines have more than 20 HP: stim to fight harder",
}

SQUAD_INSTRUCTIONS = (
    "You control a squad of 12 Terran Marines fighting Zerg Banelings and Zerglings. "
    "Goal: kill as many Zerg as possible while keeping at least one Marine alive. "
    "Score = Zerg killed if any Marine survives, otherwise 0. "
    "Running out the 60 second clock is a failure. "
    "Marines only deal damage while attacking. "
    "Pick the squad action that best serves this goal. "
    "Each action runs for 0.5 seconds before the next decision."
)

STATE_RULES = (
    "Banelings explode on contact and deal splash damage to every marine nearby; "
    "clumped marines die together. Marines have range 5 and outrange zerglings. "
    "Stim is strong but costs 10 HP per marine. "
    "Marines do not shoot while moving. The spread action steps away and then keeps attacking."
)

# Per-Marine control (episode 003): one question per living Marine, one call.
KITE_STEP = 1.5
SPLIT_STEP = 1.5
MARINE_RETREAT_DISTANCE = 3.0
MARINE_RETREAT_MAX_HP = 15

MARINE_ACTIONS = {
    "kite": "a baneling is within 3 cells of this marine: step away from it, then keep attacking",
    "split": "another marine is within 1 cell and a baneling is within 6 cells: step away from that marine, then keep attacking",
    "attack": "no baneling is within 3 cells of this marine: attack the nearest enemy",
    "stim": "enemies are within 6 cells, this marine is not stimmed and has more than 20 HP: stim",
    "retreat": "this marine has 15 HP or less and enemies are close: fall back",
}

MARINE_INSTRUCTIONS_TEMPLATE = (
    "Best action for the marine with id {tag}, given the squad goal: kill as many Zerg as possible "
    "while keeping at least one Marine alive. Each action runs for 0.5 seconds."
)
