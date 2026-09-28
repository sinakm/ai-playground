"""Scenario and runtime constants for the combat arena."""

MAP_NAME = "Flat128"
CENTER = (64.0, 64.0)

MARINE_COUNT = 12
BANELING_COUNT = 10
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
    "stim": "Enemies are within 8 cells and no marine is stimmed yet: stim now; stim adds about 50% damage output and is worth it once per fight",
}

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
    "stim": "this marine is not stimmed, has more than 20 HP and enemies are within 8 cells: stim now; stim adds about 50% damage output and pays off in almost every fight",
    "retreat": "this marine has 15 HP or less and enemies are close: fall back",
    "focus_bane": "the plan is focus_banes, or a baneling is about to reach a teammate: shoot the priority baneling",
    "cover_ally": "a teammate within 4 cells has 15 HP or less and is under attack: shoot the enemy closest to that teammate",
    "bait": "the plan is bait_and_split and you are the marine closest to the banelings: run away from the squad to pull banelings after you",
}

# jev_marine keeps the episode 003b question set: the first five actions only.
JEV_MARINE_ACTIONS = {a: MARINE_ACTIONS[a] for a in ("kite", "split", "attack", "stim", "retreat")}

MARINE_INSTRUCTIONS_TEMPLATE = (
    "Best action for the marine with id {tag}, given the squad goal: kill as many Zerg as possible "
    "while keeping at least one Marine alive. Each action runs for 0.5 seconds."
)

# Squad commander (episode 004): call 1 picks a plan and a priority Baneling,
# call 2 asks every Marine with the plan and a shared blackboard in the state.
BAIT_DISTANCE = 5.0
COVER_RADIUS = 4.0
LOW_HP = 15
PRIORITY_TARGET_CANDIDATES = 5
TEAMMATES_ON_BLACKBOARD = 3
BANELING_NEAR_TEAMMATE = 3.0
FOCUS_BANE_CAP = 4
REFLEX_KITE_DISTANCE = 2.5
FOCUS_RANGE = 5.0

SQUAD_PLANS = {
    "focus_banes": "banelings are within 6 cells of the squad: everyone shoots the priority baneling",
    "bait_and_split": "banelings are grouped and heading at a clumped squad: one marine baits, the rest spread",
    "pre_split": "banelings are 4 to 8 cells from the squad and marines are clumped: spread out before they arrive",
    "hold_and_shoot": "no baneling within 6 cells: hold ground and shoot the nearest enemy",
    "fall_back": "more than half the squad is under 20 HP and banelings are close: pull back together",
}

COMMANDER_PLAN_INSTRUCTIONS = (
    "You command a squad of 12 Terran Marines fighting Zerg Banelings and Zerglings. "
    "Goal: kill as many Zerg as possible while keeping at least one Marine alive. "
    "Score = Zerg killed if any Marine survives, otherwise 0. "
    "Pick the squad plan for the next 0.5 seconds; every Marine will see it."
)

STIM_NOW_INSTRUCTIONS = (
    "Should the whole squad stim now? Yes when enemies are within 8 cells, "
    "most marines are unstimmed and above 20 HP."
)

# Stim is the commander's call (stim_now), so soldiers under the commander cannot pick it.
COMMANDER_MARINE_ACTIONS = {a: t for a, t in MARINE_ACTIONS.items() if a != "stim"}

COMMANDER_TARGET_INSTRUCTIONS = (
    "Pick the baneling the squad should kill first: the one that threatens the most marines soonest. "
    "Marines that choose focus_bane will shoot it."
)

COMMANDER_MARINE_INSTRUCTIONS_TEMPLATE = (
    "Best action for the marine with id {tag}, given the squad goal: kill as many Zerg as possible "
    "while keeping at least one Marine alive. Your squad_plan and role facts are in the state; follow the plan "
    "unless your own situation clearly needs another action. Use your teammates on the blackboard "
    "to decide whether to cover an ally. Each action runs for 0.5 seconds."
)
